import json
import base64
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name="ap-southeast-1")
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def generate_questions():
    body = request.get_json(force=True)

    subject = body.get("subject", "")
    focus_topics = body.get("focus_topics", "")
    difficulty = body.get("difficulty", "Medium")
    num_questions = body.get("num_questions", 10)
    question_type = body.get("question_type", "MCQ")
    file_data = body.get("file_data", "")
    file_mime = body.get("file_mime", "")

    prompt_text = (
        f"You are an expert educator and exam designer. Using the content from the uploaded lecture material below, "
        f"generate exactly {num_questions} exam questions of the type: {question_type}, "
        f"at a {difficulty} difficulty level for the subject: {subject}\n\n"
    )

    if focus_topics:
        prompt_text += f"If specific focus topics were provided, prioritize those: {focus_topics}\n\n"

    prompt_text += (
        "**IMPORTANT FORMAT:**\n\n"
        f"1. Display questions ONE BY ONE, numbered sequentially:\n"
        f"- Question 1 of {num_questions}\n"
        f"- Question text\n"
        f"- A line\n"
        f"- Question 2 of {num_questions}\n"
        f"- Question text\n"
        f"- line\n"
        f"- Continue for all questions\n\n"
        "After all questions, provide a complete Answer Key section with:\n"
        "- For MCQ: the correct option letter and a brief explanation\n"
        "- For Short Answer: a model answer (2-4 sentences)\n"
        "- For Essay: key points that should be covered\n"
    )

    # Build content blocks
    content = []

    if file_data and file_mime:
        if file_mime.startswith("image/"):
            # Image block
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": file_mime,
                    "data": file_data,
                },
            })
            content.append({
                "type": "text",
                "text": prompt_text + "\n\nLecture Content:\n[See attached image above]",
            })
        else:
            # Document block (PDF, DOCX, PPTX, etc.)
            content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": file_mime,
                    "data": file_data,
                },
            })
            content.append({
                "type": "text",
                "text": prompt_text + "\n\nLecture Content:\n[See attached document above]",
            })
    else:
        content.append({
            "type": "text",
            "text": prompt_text + "\n\nLecture Content:\n[No file uploaded — generate questions based on the subject and focus topics provided]",
        })

    messages = [{"role": "user", "content": content}]

    bedrock_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": messages,
    }

    def generate():
        try:
            response = BEDROCK_CLIENT.invoke_model_with_response_stream(
                modelId=MODEL_ID,
                body=json.dumps(bedrock_body),
                contentType="application/json",
                accept="application/json",
            )
            for event in response["body"]:
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk["bytes"].decode("utf-8"))
                    if chunk_data.get("type") == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
        except Exception as e:
            yield f"\n\n[ERROR] {str(e)}"

    resp = Response(
        stream_with_context(generate()),
        content_type="text/plain; charset=utf-8",
        headers=CORS_HEADERS,
    )
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
