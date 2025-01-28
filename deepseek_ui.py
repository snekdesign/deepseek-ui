import contextlib
import io
import logging
from typing import cast
import warnings

with contextlib.redirect_stdout(None), warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from janus.models import MultiModalityCausalLM, VLChatProcessor  # pyright: ignore[reportMissingTypeStubs]
from janus.models.vq_model import VQModel  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import ollama
import PIL.Image
import streamlit as st
import torch
from transformers import AutoModelForCausalLM  # pyright: ignore[reportMissingTypeStubs]
from transformers.modeling_outputs import BaseModelOutputWithPast  # pyright: ignore[reportMissingTypeStubs]

_MODEL_PATH = 'data'

st.set_page_config('DeepSeek')


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 6,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    tokens = torch.tensor(
        vl_chat_processor.tokenizer.encode(prompt),  # pyright: ignore[reportUnknownMemberType]
        dtype=torch.int,
        device='cuda',
    ).repeat(parallel_size*2, 1)

    assert vl_chat_processor.pad_id is not None
    tokens[1::2, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)  # pyright: ignore[reportUnknownVariableType]

    generated_tokens = torch.empty(
        size=(parallel_size, image_token_num_per_image),
        dtype=torch.int,
        device='cuda',
    )

    past_key_values = None
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        assert isinstance(outputs, BaseModelOutputWithPast)
        past_key_values = outputs.past_key_values

        logits = mmgpt.gen_head(outputs.last_hidden_state[:, -1])
        logit_cond = logits[::2]
        logit_uncond = logits[1::2]

        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits/temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)],
            dim=1,
        ).view(-1)
        img_embeds: torch.Tensor = mmgpt.prepare_gen_img_embeds(next_token)  # pyright: ignore[reportArgumentType]
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    assert isinstance(mmgpt.gen_vision_model, VQModel)
    dec: torch.Tensor = mmgpt.gen_vision_model.decode_code(  # pyright: ignore[reportUnknownMemberType]
        generated_tokens,
        shape=(parallel_size, 8, img_size//patch_size, img_size//patch_size),
    )
    response: list[bytes] = []
    for obj in np.asarray(
        dec.add_(1)
           .mul_(255/2)
           .clamp_(0, 255)
           .to(torch.uint8)
           .cpu()
           .permute(0, 2, 3, 1),
    ):
        with io.BytesIO() as f:
            PIL.Image.fromarray(obj).save(f, 'JPEG2000')
            response.append(f.getvalue())
    return response


@st.cache_resource
def models():
    for name in (
        'transformers.models.auto.image_processing_auto',
        'transformers.models.llama.tokenization_llama_fast',
        'transformers.processing_utils',
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    vl_chat_processor = VLChatProcessor.from_pretrained(_MODEL_PATH)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert isinstance(vl_chat_processor, VLChatProcessor)

    mmgpt = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        _MODEL_PATH,
        trust_remote_code=True,
    )
    mmgpt = cast(torch.nn.Module, mmgpt).to(torch.bfloat16).cuda().eval()
    assert isinstance(mmgpt, MultiModalityCausalLM)

    return (mmgpt, vl_chat_processor)


messages: list[ollama.Message] = st.session_state.setdefault('messages', [])
for m in messages:
    with st.chat_message(m.role):
        if m.content:
            st.markdown(m.content, m.role == 'assistant')
        elif m.images:
            st.image([image.value for image in m.images])
        else:
            assert False, 'unreachable'
if prompt := st.chat_input('给 DeepSeek 发送消息'):
    st.chat_message('user').markdown(prompt)
    messages.append(ollama.Message(role='user', content=prompt))
    with st.chat_message('assistant'):
        mmgpt, vl_chat_processor = models()
        sft_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts([
            {'role': '<|User|>', 'content': prompt},
            {'role': '<|Assistant|>', 'content': ''},
        ])
        response = generate(
            mmgpt,
            vl_chat_processor,
            prompt=sft_prompt+vl_chat_processor.image_start_tag,
        )
        st.image(response)
    messages.append(ollama.Message(
        role='assistant',
        images=[ollama.Image(value=value) for value in response],
    ))
