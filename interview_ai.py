import os
import time
import re
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from rich import print
from openai import OpenAI

# ================= CONFIG =================
SAMPLE_RATE = 16000
DURATION = 5

# ================= LOAD WHISPER =================
print("[cyan]🚀 Loading Whisper model...[/cyan]")
whisper_model = WhisperModel("tiny", compute_type="int8")

# ================= GROQ SETUP =================
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not set in environment")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

print("[green]✅ Interview Assistant Ready![/green]\n")


# ================= AUDIO RECORD =================
def record_audio():
    print("[cyan]🎤 Listening...[/cyan]")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    return audio.flatten()


# ================= STREAMING Q&A =================
def ask_ai(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": "You are a technical interview assistant. Give short, precise, professional answers."},
                {"role": "user", "content": question}
            ],
            temperature=0.4,
            stream=True
        )

        full_response = ""
        print("[green]🤖 AI:[/green] ", end="", flush=True)

        for chunk in response:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full_response += text

        print("\n")
        return full_response.strip()

    except Exception as e:
        return f"AI error: {e}"


# ================= SAVE LOG =================
def save_log(question: str, answer: str):
    with open("interview_log.txt", "a", encoding="utf-8") as f:
        f.write("=====================================\n")
        f.write(f"Q: {question}\n\n")
        f.write(f"A: {answer}\n")
        f.write("=====================================\n\n")


# ================= FOLLOW-UP ENGINE =================
def generate_follow_up(question, answer):
    prompt = f"""
    Original Question:
    {question}

    Candidate Answer:
    {answer}

    If weak → ask deeper clarification.
    If strong → ask advanced follow-up.

    Only output ONE follow-up question.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a strict senior interviewer."},
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content.strip()


# ================= MOCK INTERVIEW =================
def mock_interview():
    print("\n[magenta]🧠 Mock Interview Mode Started[/magenta]\n")

    domain = input("Choose domain (python / dsa / ml / system): ").strip()
    difficulty = input("Choose difficulty (easy / medium / hard): ").strip()
    total_rounds = int(input("How many rounds? "))

    scores = []

    for round_number in range(1, total_rounds + 1):

        print(f"\n[blue]========== ROUND {round_number} ==========[/blue]\n")

        question_prompt = f"""
        Ask ONE {difficulty} level technical interview question in {domain}.
        Only ask the question.
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a technical interviewer."},
                {"role": "user", "content": question_prompt}
            ],
        )

        interview_question = response.choices[0].message.content.strip()

        print(f"[cyan]👨‍💼 Interviewer:[/cyan] {interview_question}\n")
        print("[yellow]🎤 Speak your answer now...[/yellow]")

        audio = record_audio()
        segments, _ = whisper_model.transcribe(audio)
        user_answer = " ".join(segment.text for segment in segments).strip()

        print(f"\n[yellow]🗣 Your Answer:[/yellow] {user_answer}\n")

        evaluation_prompt = f"""
        Evaluate this candidate strictly.

        Question:
        {interview_question}

        Candidate Answer:
        {user_answer}

        Format:
        Strengths:
        Weaknesses:
        What was missing:
        Score: X/10
        """

        evaluation = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a strict senior technical interviewer."},
                {"role": "user", "content": evaluation_prompt}
            ],
        )

        feedback = evaluation.choices[0].message.content.strip()

        print(f"[green]📊 Evaluation:[/green]\n{feedback}\n")

        match = re.search(r'(\d+)/10', feedback)
        if match:
            scores.append(int(match.group(1)))

        # Follow-Up
        follow_up = generate_follow_up(interview_question, user_answer)

        print(f"[red]🔎 Follow-Up Question:[/red] {follow_up}\n")
        print("[yellow]🎤 Answer follow-up...[/yellow]")

        audio = record_audio()
        segments, _ = whisper_model.transcribe(audio)
        follow_answer = " ".join(segment.text for segment in segments).strip()

        print(f"\n[yellow]🗣 Follow-Up Answer:[/yellow] {follow_answer}\n")

        save_log(interview_question,
                 f"Main Answer:\n{user_answer}\n\nEvaluation:\n{feedback}\n\nFollow-Up:\n{follow_up}\nAnswer:\n{follow_answer}")

    # Final Report
    if scores:
        avg = sum(scores) / len(scores)

        print("\n[magenta]========== FINAL REPORT ==========[/magenta]\n")
        print(f"Average Score: {avg:.2f}/10")

        if avg >= 8:
            print("[green]Status: Interview Ready 🔥[/green]")
        elif avg >= 5:
            print("[yellow]Status: Improving ⚡[/yellow]")
        else:
            print("[red]Status: Needs Strong Preparation 🚨[/red]")


# ================= MAIN LOOP =================
while True:
    print("\nSelect Mode:")
    print("1. Normal Q&A Mode")
    print("2. Mock Interview Mode")
    print("3. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        audio = record_audio()
        segments, _ = whisper_model.transcribe(audio)
        text = " ".join(segment.text for segment in segments).strip()

        if text:
            print(f"[yellow]🗣 You said:[/yellow] {text}")
            answer = ask_ai(text)
            save_log(text, answer)

    elif choice == "2":
        mock_interview()

    elif choice == "3":
        print("[red]Exiting InterviewAI...[/red]")
        break
