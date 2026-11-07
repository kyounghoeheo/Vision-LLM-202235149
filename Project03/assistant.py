import json
import subprocess
from pathlib import Path

from llama_cpp import Llama


# 05 강의의 Gemma 및 06 강의의 Whisper/Piper 함수 구조 유지
PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
RECORD_PATH = PROJECT_DIR / "data/lost_items.json"

GEMMA_MODEL_PATH = ROOT_DIR / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
CONTEXT_WINDOW = 2048
MAX_TOKENS = 150

MIC_DEVICE = "plughw:3,0"  # Jetson에서 'arecord -l'로 마이크 번호 확인
AUDIO_FILE = PROJECT_DIR / "data/input.wav"
WHISPER_PATH = ROOT_DIR / "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = ROOT_DIR / "whisper.cpp/models/ggml-base.bin"
RECORD_SECONDS = 5

PIPER_PYTHON = ROOT_DIR / ".piper_venv/bin/python"
PIPER_MODEL = ROOT_DIR / "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = PROJECT_DIR / "data/response.wav"


def speech_to_text():
    print("말씀을 시작해 주세요.")
    subprocess.run(
        [
            "pasuspender", "--",
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(RECORD_SECONDS),
            str(AUDIO_FILE),
        ],
        check=True,
    )

    result = subprocess.run(
        [
            str(WHISPER_PATH),
            "-m", str(WHISPER_MODEL),
            "-f", str(AUDIO_FILE),
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def text_to_speech(text):
    subprocess.run(
        [
            str(PIPER_PYTHON),
            "-m", "piper",
            "-m", str(PIPER_MODEL),
            "-f", str(OUTPUT_FILE),
            "--", text,
        ],
        check=True,
    )
    subprocess.run(["aplay", str(OUTPUT_FILE)], check=True)


def records_to_text():
    if not RECORD_PATH.is_file():
        return "접수된 분실물이 없습니다."

    with open(RECORD_PATH, "r", encoding="utf-8") as file:
        records = json.load(file)

    if len(records) == 0:
        return "접수된 분실물이 없습니다."

    sentences = []
    for item in records:
        sentence = (
            f"접수번호 {item['id']}: {item['class']}, "
            f"분류 {item['category']}, "
            f"안내 보관함 {item['recommended_box']}, "
            f"접수시각 {item['registered_at']}"
        )
        sentences.append(sentence)

    return "\n".join(sentences)


def main():
    PROJECT_DIR.joinpath("data").mkdir(exist_ok=True)
    if not GEMMA_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Gemma 모델 파일이 없습니다: {GEMMA_MODEL_PATH}")

    llm = Llama(
        model_path=str(GEMMA_MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

    print("분실물 음성 안내 시스템")
    print("Enter: 질문 시작 / q: 종료")

    while True:
        command = input("\n질문을 시작하려면 Enter를 누르세요 (종료 q): ").strip().lower()
        if command in ("q", "quit", "exit"):
            break

        try:
            question = speech_to_text()
            print("[질문]", question)

            if not question:
                print("음성이 인식되지 않았습니다. 다시 시도하세요.")
                continue

            # 06 강의: JSON -> Natural Language -> Gemma Context
            vision_text = records_to_text()
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 학교 분실물 관리 안내 도우미입니다. "
                            "아래 접수 기록에 있는 정보만 바탕으로 대답하세요. "
                            "접수되지 않은 물건이나 실제 소유자를 추측하지 마세요. "
                            "보관함은 권장 위치일 뿐, 실제 보관 완료를 뜻하지 않습니다. "
                            "정보가 없으면 없다고 답하세요. 한국어 두 문장 이내로 간단히 답변하세요.\n\n"
                            "접수 기록:\n" + vision_text
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.2,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            print("[Gemma]", answer)

            if answer:
                text_to_speech(answer)
        except (OSError, subprocess.CalledProcessError) as error:
            print("음성 처리 오류:", error)
            print("마이크 번호, Whisper 경로, Piper 환경을 확인하세요.")


if __name__ == "__main__":
    main()
