from collections.abc import Generator
from collections.abc import Sequence
import contextlib
import io
import logging
import time
from typing import Any
from typing import cast
from typing import List
from typing import TYPE_CHECKING
import warnings

with contextlib.redirect_stdout(None), warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from janus.models import MultiModalityCausalLM, VLChatProcessor  # pyright: ignore[reportMissingTypeStubs]
from janus.models.processing_vlm import BatchedVLChatProcessorOutput  # pyright: ignore[reportMissingTypeStubs]
from janus.models.vq_model import VQModel  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import ollama
import PIL.Image
import streamlit as st
if TYPE_CHECKING:
    from streamlit.runtime.caching.cache_utils import CachedFunc
import streamlit_scroll_to_top  # pyright: ignore[reportMissingTypeStubs]
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
        vl_chat_processor.tokenizer.encode(prompt),  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
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
    visual_img: List[bytes] = []
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
            visual_img.append(f.getvalue())
    return visual_img


@st.cache_resource(show_spinner=False)
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
    assert isinstance(mmgpt, torch.nn.Module)
    mmgpt = mmgpt.to(torch.bfloat16).cuda().eval()
    assert isinstance(mmgpt, MultiModalityCausalLM)

    return (mmgpt, vl_chat_processor)


def on_submit():
    # https://discuss.streamlit.io/t/disable-chat-input-while-chat-bot-is-responding/70507
    st.session_state.disabled = True


def prepare_inputs(
    vl_chat_processor: VLChatProcessor,
    images: Sequence[ollama.Image],
    prompt: str,
) -> BatchedVLChatProcessorOutput:
    value = images[0].value
    assert isinstance(value, bytes)
    with io.BytesIO(value) as f:
        image = PIL.Image.open(f).convert('RGB')
    return vl_chat_processor(
        conversations=[
            {'role': '<|User|>', 'content': f'<image_placeholder>\n{prompt}'},
            {'role': '<|Assistant|>', 'content': ''},
        ],
        images=[image],
    )


def response(prompt: str):
    with st.chat_message('assistant'):
        if st.session_state.model == 'janus-pro:1b':
            mmgpt, vl_chat_processor = models()
            if images := messages[0].images:
                inputs = prepare_inputs(
                    vl_chat_processor,
                    images,
                    prompt,
                ).to(mmgpt.device)  # pyright: ignore[reportUnknownMemberType]
                tokenizer = vl_chat_processor.tokenizer
                outputs = mmgpt.language_model.generate(  # pyright: ignore[reportUnknownMemberType]
                    inputs_embeds=mmgpt.prepare_inputs_embeds(**inputs),  # pyright: ignore[reportUnknownMemberType]
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,  # pyright: ignore[reportUnknownMemberType]
                    bos_token_id=tokenizer.bos_token_id,  # pyright: ignore[reportUnknownMemberType]
                    eos_token_id=tokenizer.eos_token_id,  # pyright: ignore[reportUnknownMemberType]
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )
                assert isinstance(outputs, torch.Tensor)
                content = tokenizer.decode(outputs[0], skip_special_tokens=True)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
                return ollama.Message(role='assistant', content=content)
            sft_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts([
                {'role': '<|User|>', 'content': prompt},
                {'role': '<|Assistant|>', 'content': ''},
            ])
            visual_img = generate(
                mmgpt,
                vl_chat_processor,
                prompt=sft_prompt+vl_chat_processor.image_start_tag,
            )
            return ollama.Message(
                role='assistant',
                images=[ollama.Image(value=value) for value in visual_img],
            )
        stream = Stream()
        if stream.thinking:
            label = '已深度思考'
            with st.status('思考中...', expanded=True) as status:
                with st.empty():
                    while stream.thinking:
                        if (
                            (thoughts := st.write_stream(stream))
                            and not cast(str, thoughts).isspace()
                        ):
                            try:
                                timing = think[len(messages)+1]
                            except KeyError:
                                stream.close()
                                stream = Stream()
                            else:
                                label += f' (用时 {timing} 秒)'
                                break
                        else:
                            st.markdown('')
                            break
                    else:
                        st.markdown('')
                        thoughts = []
                status.update(label=label, expanded=False, state='complete')
        else:
            thoughts = []
        content = st.write_stream(stream)
        assert isinstance(content, str)
        if isinstance(thoughts, str):
            content = f'<think>{thoughts}</think>{content}'
        return ollama.Message(role='assistant', content=content)


class Stream:
    def __init__(self):
        self.thinking = False
        self._responses = responses = ollama.chat(  # pyright: ignore[reportUnknownMemberType]
            st.session_state.model, messages, stream=True)
        self._tic = time.monotonic()
        for r in responses:
            if chunk := r.message.content:
                if chunk.startswith('<think>'):
                    self.thinking = True
                    self._initial = chunk[7:]
                else:
                    self._initial = chunk
                return
        self._initial = None

    def __iter__(self) -> Generator[str, Any, None]:
        with contextlib.closing(self._iter()) as it:
            if self.thinking:
                for chunk in it:
                    if '</think>' in chunk:
                        toc = time.monotonic()
                        think[len(messages) + 1] = round(toc - self._tic)
                        self.thinking = False
                        chunk, self._initial = chunk.split('</think>', 1)
                        yield chunk
                        return
                    yield chunk
            else:
                yield from it
        self._initial = None

    def close(self):
        if isinstance(self._responses, Generator):
            self._responses.close()  # pyright: ignore[reportUnknownMemberType]

    def _iter(self) -> Generator[str, Any, None]:
        if self._initial is None:
            return
        yield self._initial
        for r in self._responses:
            if chunk := r.message.content:
                yield chunk


def unload_ollama():
    # https://github.com/ollama/ollama/blob/main/docs/api.md#unload-a-model-1
    response = ollama.chat('deepseek-r1:8b', messages=[], keep_alive=0)  # pyright: ignore[reportUnknownMemberType]
    if not (
        response.done and response.done_reason == 'unload'
        and response.model == 'deepseek-r1:8b'
    ):
        st.json(response.model_dump(mode='json'))
        st.stop()


messages: List[ollama.Message] = st.session_state.setdefault('messages', [])
if messages:
    disabled = st.session_state.disabled
    think: dict[int, int] = st.session_state.setdefault('think', {})
    n = len(messages)
    for i, m in enumerate(messages, 1):
        if i == n and not disabled:
            streamlit_scroll_to_top.scroll_to_here()  # pyright: ignore[reportArgumentType]
        with st.chat_message(m.role):
            if content := m.content:
                if (
                    (unsafe_allow_html := m.role == 'assistant')
                    and content.startswith('<think>')
                    and (j := content.find('</think>')) > 7
                    and not (thoughts := content[7:j]).isspace()
                ):
                    with st.status('思考中...') as status:
                        st.markdown(thoughts)
                        status.update(
                            label=f'已深度思考 (用时 {think[i]} 秒)',
                            state='complete',
                        )
                    st.markdown(content[j+8 :])
                else:
                    st.markdown(content, unsafe_allow_html)
            elif m.images:
                st.image([image.value for image in m.images])
            else:
                assert False, 'unreachable'
    if messages[-1].role == 'user':
        if prompt := messages[-1].content:
            messages.append(response(prompt))
            st.session_state.disabled = False
            st.rerun()
    if prompt := st.chat_input(
        '给 DeepSeek 发送消息',
        disabled=disabled,
        on_submit=on_submit,
    ):
        streamlit_scroll_to_top.scroll_to_here()  # pyright: ignore[reportArgumentType]
        st.chat_message('user').markdown(prompt)
        messages.append(ollama.Message(role='user', content=prompt))
        messages.append(response(prompt))
        st.session_state.disabled = False
        st.rerun()
else:
    tab1, tab2 = st.tabs(['对话/图像识别', '图像生成'])
    prompt2 = tab1.chat_input(
        '给 DeepSeek 发送消息',
        accept_file=True,
        on_submit=on_submit,
    )
    if prompt := tab2.chat_input(
        '给 DeepSeek 发送消息',
        on_submit=on_submit,
    ):
        unload_ollama()
        st.session_state.model = 'janus-pro:1b'
        messages.append(ollama.Message(role='user', content=prompt))
        st.rerun()
    elif prompt2:
        if prompt2.files:
            try:
                with PIL.Image.open(prompt2.files[0]):
                    pass
            except PIL.UnidentifiedImageError as e:
                tab1.exception(e.with_traceback(None))
                st.stop()
            unload_ollama()
            st.session_state.model = 'janus-pro:1b'
            message = ollama.Message(
                role='user',
                images=[ollama.Image(value=prompt2.files[0].getvalue())],
            )
            messages.append(message)
            if prompt2.text:
                message = ollama.Message(role='user', content=prompt2.text)
                messages.append(message)
            else:
                st.session_state.disabled = False
            st.rerun()
        elif prompt2.text:
            cast('CachedFunc', models).clear()  # pyright: ignore[reportUnknownMemberType]
            torch.cuda.empty_cache()
            st.session_state.model = 'deepseek-r1:8b'
            messages.append(ollama.Message(role='user', content=prompt2.text))
            st.rerun()
