import base64
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
AGENT_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"
MAX_STEPS = 6

PARTS = {
    "water pump mechanical seal": {
        "name": "water pump mechanical seal",
        "quantity": 1,
        "price_range": "$25-$60",
        "where": ["Partzilla", "RevZilla"],
    },
    "clutch cover gasket": {
        "name": "clutch cover gasket",
        "quantity": 1,
        "price_range": "$15-$40",
        "where": ["Partzilla", "RevZilla"],
    },
}

SYSTEM = """You are a diagnostic agent for a 2013-2018 Honda CB500F.
You may act only by calling the provided tools.
Call describe_image before search_repair_notes.
Call search_repair_notes before submit_diagnosis.
Cite only chunk ids that search_repair_notes returned in this session.
If the photo is unusable, or the retrieved chunks do not support a cause, call request_more_info and do not invent a forum post.
Do not state a torque value.
describe_image reports only what is visible. The cause comes from retrieved chunks."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "describe_image",
            "description": "Describe only what is visible in the case photo. Do not give a cause.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repair_notes",
            "description": "Embed the photo description, symptoms, and bike, then return the closest forum chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "Optional short phrase about the visible part. The case symptoms and bike are added automatically.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_parts",
            "description": "Look up a part in the local parts table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_name": {
                        "type": "string",
                        "enum": list(PARTS),
                    }
                },
                "required": ["part_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_more_info",
            "description": "Stop and ask for a better photo or clearer symptoms.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_diagnosis",
            "description": "Submit the diagnosis. Every chunk id must come from this session's search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "causes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "likelihood": {"type": "string", "enum": ["high", "medium", "low"]},
                                "chunk_id": {"type": "string"},
                            },
                            "required": ["name", "likelihood", "chunk_id"],
                            "additionalProperties": False,
                        },
                    },
                    "repair_suggestion": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short suggestions supported by the retrieved chunks. Not a complete repair procedure.",
                    },
                    "ride": {"type": "string", "enum": ["ride", "limited", "do_not_ride"]},
                    "chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["causes", "repair_suggestion", "ride", "chunk_ids"],
                "additionalProperties": False,
            },
        },
    },
]


def load_env_file(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_case(path):
    case = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        case[key.strip()] = value.strip()
    required = ["year", "model", "symptoms", "mileage", "photo"]
    missing = [key for key in required if key not in case]
    if missing:
        raise SystemExit(f"Case file is missing: {', '.join(missing)}")
    return case


def load_chunks(path):
    chunks = json.loads(path.read_text() or "[]")
    if not chunks:
        raise SystemExit("knowledge-base/chunks.json has no chunks.")
    for chunk in chunks:
        for key in ("id", "text", "source"):
            if key not in chunk or chunk[key] in ("", "原帖地址"):
                raise SystemExit(f"Chunk {chunk.get('id', '?')} needs a real {key}.")
    return chunks


def cosine(left, right):    # consine 相似度， 数值越大越相似
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return dot / (left_norm * right_norm)


def embed_texts(client, texts):
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def describe_image(client, photo_path):
    payload = base64.standard_b64encode(photo_path.read_bytes()).decode()
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe only what is visible on this motorcycle part. "
                            "Mention location, residue, wear, cracks, and holes. "
                            "Do not name a cause or a repair."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


def search_repair_notes(client, arguments, state, chunks, vectors):
    query = " ".join(
        part
        for part in (
            state.get("photo_description") or "",
            state["symptoms"],
            state["year"],
            state["model"],
            arguments.get("focus") or "",
        )
        if part
    )
    query_vector = embed_texts(client, [query])[0]
    ranked = sorted(
        (
            {
                "id": chunk["id"],
                "part": chunk.get("part", ""),
                "text": chunk["text"],
                "source": chunk["source"],
                "score": round(cosine(query_vector, vector), 4),
            }
            for chunk, vector in zip(chunks, vectors)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    found = ranked[:3]
    state["retrieved"] = found
    return {"query": query, "chunks": found}


def lookup_parts(arguments):
    part = PARTS.get(arguments.get("part_name"))
    if part is None:
        return {"found": False, "part_name": arguments.get("part_name")}
    return {"found": True, "part": part}


def submit_diagnosis(arguments, state):
    retrieved_ids = {item["id"] for item in state["retrieved"]}
    cited = arguments.get("chunk_ids") or []
    if not retrieved_ids:
        return {
            "accepted": False,
            "error": "search_repair_notes has not been called. Search before submitting.",
        }, False
    unknown = [chunk_id for chunk_id in cited if chunk_id not in retrieved_ids]
    if unknown or not cited:
        return {
            "accepted": False,
            "error": "Cite only chunk ids returned by this search.",
            "retrieved_ids": sorted(retrieved_ids),
        }, False
    return {"accepted": True, "diagnosis": arguments}, True


def run(case_path):
    load_env_file(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Put OPENAI_API_KEY in .env or the environment.")

    case = load_case(case_path)
    chunks = load_chunks(ROOT / "knowledge-base" / "chunks.json")
    photo_path = ROOT / case["photo"]
    if not photo_path.exists():
        raise SystemExit(f"Photo not found: {photo_path}")

    client = OpenAI()
    vectors = embed_texts(client, [chunk["text"] for chunk in chunks])
    state = {
        "year": case["year"],
        "model": case["model"],
        "symptoms": case["symptoms"],
        "mileage": case["mileage"],
        "photo_description": None,
        "retrieved": [],
    }
    trace = {
        "case_file": str(case_path.relative_to(ROOT)),
        "gold_fault_not_shown_to_model": case.get("fault"),
        "steps": [],
        "final": None,
    }
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Year: {case['year']}\n"
                f"Model: {case['model']}\n"
                f"Mileage: {case['mileage']}\n"
                f"Symptoms: {case['symptoms']}\n"
                "The photo is available only through describe_image."
            ),
        },
    ]

    for step_number in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            trace["steps"].append(
                {"step": step_number, "tool": None, "text": message.content}
            )
            trace["final"] = {"stopped": "model returned text instead of a tool"}
            break

        finished = False
        for call in message.tool_calls:
            arguments = json.loads(call.function.arguments or "{}")
            if call.function.name == "describe_image":
                description = describe_image(client, photo_path)
                state["photo_description"] = description
                result, done = {"description": description}, False
            elif call.function.name == "search_repair_notes":
                result, done = search_repair_notes(
                    client, arguments, state, chunks, vectors
                ), False
            elif call.function.name == "lookup_parts":
                result, done = lookup_parts(arguments), False
            elif call.function.name == "request_more_info":
                result, done = {"message": arguments.get("message", "")}, True
            elif call.function.name == "submit_diagnosis":
                result, done = submit_diagnosis(arguments, state)
            else:
                result, done = {"error": f"Unknown tool {call.function.name}"}, False

            trace["steps"].append(
                {
                    "step": step_number,
                    "tool": call.function.name,
                    "arguments": arguments,
                    "result": result,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )
            if done:
                trace["final"] = result
                finished = True
        if finished:
            break
    else:
        trace["final"] = {"stopped": "reached 6 steps without a final action"}

    trace_path = ROOT / "traces" / f"{case_path.stem}.json"
    trace_path.parent.mkdir(exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
    print(json.dumps(trace["final"], indent=2, ensure_ascii=False))
    print(f"Trace written to {trace_path.relative_to(ROOT)}")


if __name__ == "__main__":
    chosen = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cases" / "water-pump.txt"
    if not chosen.is_absolute():
        chosen = ROOT / chosen
    run(chosen)
