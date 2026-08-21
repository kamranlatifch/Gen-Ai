"""A text editor the model can drive: read, create, and edit files.

Every tool so far only fetched information - the worst a wrong answer could do
was be wrong. These tools write to disk. That changes what matters:

    Sandbox     the model picks the paths, so it must not be able to reach
                outside one directory. _resolve() enforces that.
    Uniqueness  str_replace refuses unless the old text appears exactly once,
                so a vague edit fails loudly instead of silently hitting the
                wrong line.
    Undo        every write stashes the previous contents first, so a bad
                edit is recoverable.

This is roughly how coding assistants edit code: the model never touches the
filesystem, it asks for an edit and your code decides whether to allow it.
"""

import json
from pathlib import Path

from openai import RateLimitError

from generate_datasets import GROQ_MODEL, MODEL, client, groq_client

# Everything the model touches lives here. Created on first use.
SANDBOX = Path(__file__).parent / "editor_workspace"

# path -> stack of previous contents, so undo_edit can step back.
_history = {}


def _resolve(path):
    """Turn a model-supplied path into a real path inside the sandbox.

    The model chooses these strings, so treat them as untrusted. Without the
    check below, "../../.env" or "/etc/passwd" would be perfectly valid input.
    resolve() collapses any ".." before we compare, so the check cannot be
    tricked by a path that only looks safe.
    """
    SANDBOX.mkdir(exist_ok=True)
    full_path = (SANDBOX / path).resolve()
    if not full_path.is_relative_to(SANDBOX.resolve()):
        raise ValueError(f"Path {path!r} is outside the sandbox.")
    return full_path


def _save_for_undo(full_path):
    previous = full_path.read_text(encoding="utf-8") if full_path.exists() else None
    _history.setdefault(str(full_path), []).append(previous)


# --------------------------------------------------------------------------
# The tools
# --------------------------------------------------------------------------


def list_files():
    """List the files in the workspace."""
    SANDBOX.mkdir(exist_ok=True)
    names = sorted(p.name for p in SANDBOX.iterdir() if p.is_file())
    return "\n".join(names) if names else "The workspace is empty."


def view_file(path, start_line=None, end_line=None):
    """Read a file, numbered by line."""
    full_path = _resolve(path)
    if not full_path.exists():
        return f"Error: {path} does not exist."

    lines = full_path.read_text(encoding="utf-8").splitlines()
    first = start_line or 1
    last = end_line or len(lines)

    # Line numbers are not decoration. They give the model a way to say
    # "insert at line 7", and they make str_replace targets easier to quote
    # back exactly.
    numbered = [f"{i}\t{lines[i - 1]}" for i in range(first, min(last, len(lines)) + 1)]
    return "\n".join(numbered) if numbered else "(file is empty)"


def create_file(path, file_text):
    """Create a file, or overwrite one that already exists."""
    full_path = _resolve(path)
    _save_for_undo(full_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(file_text, encoding="utf-8")
    return f"Wrote {len(file_text)} characters to {path}."


def str_replace(path, old_str, new_str):
    """Replace one exact piece of text with another."""
    full_path = _resolve(path)
    if not full_path.exists():
        return f"Error: {path} does not exist."

    content = full_path.read_text(encoding="utf-8")
    count = content.count(old_str)

    # Refusing on 0 and on 2+ is the whole safety design. Zero means the model
    # misremembered the file. More than one means the edit is ambiguous and we
    # would be guessing which line it meant.
    if count == 0:
        return f"Error: old_str not found in {path}. View the file and quote it exactly."
    if count > 1:
        return f"Error: old_str appears {count} times in {path}. Include more surrounding lines to make it unique."

    _save_for_undo(full_path)
    full_path.write_text(content.replace(old_str, new_str), encoding="utf-8")
    return f"Replaced 1 occurrence in {path}."


def insert_line(path, line_number, text):
    """Insert text after the given line number. Use 0 for the top of the file."""
    full_path = _resolve(path)
    if not full_path.exists():
        return f"Error: {path} does not exist."

    lines = full_path.read_text(encoding="utf-8").splitlines()
    if not 0 <= line_number <= len(lines):
        return f"Error: line {line_number} is out of range, the file has {len(lines)} lines."

    _save_for_undo(full_path)
    lines.insert(line_number, text)
    full_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"Inserted a line after line {line_number} in {path}."


def undo_edit(path):
    """Restore a file to how it was before the last edit."""
    full_path = _resolve(path)
    stack = _history.get(str(full_path))
    if not stack:
        return f"Error: no edits recorded for {path}."

    previous = stack.pop()
    if previous is None:
        full_path.unlink(missing_ok=True)
        return f"Undid creation of {path}, the file is gone again."

    full_path.write_text(previous, encoding="utf-8")
    return f"Reverted {path} to its previous contents."


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


def _schema(name, description, properties=None, required=None):
    """Small helper - these six schemas are otherwise mostly boilerplate."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


TOOLS = [
    _schema(
        "list_files",
        "List the files in the workspace. Call this first when you do not "
        "already know what files exist.",
    ),
    _schema(
        "view_file",
        "Read a file and return its contents with a line number in front of "
        "each line. Always view a file before editing it, so that the text you "
        "quote to str_replace matches the file exactly.",
        {
            "path": {"type": "string", "description": "File name, e.g. 'greet.py'."},
            "start_line": {"type": "number", "description": "First line to show. Optional."},
            "end_line": {"type": "number", "description": "Last line to show. Optional."},
        },
        ["path"],
    ),
    _schema(
        "create_file",
        "Create a new file with the given contents, replacing it if it already "
        "exists. Use this for new files, not for editing existing ones.",
        {
            "path": {"type": "string", "description": "File name, e.g. 'greet.py'."},
            "file_text": {"type": "string", "description": "The complete contents of the file."},
        },
        ["path", "file_text"],
    ),
    _schema(
        "str_replace",
        "Replace an exact piece of text in a file. old_str must appear exactly "
        "once, so include enough surrounding lines to make it unique. This is "
        "the preferred way to edit an existing file, because it changes only "
        "the part you name and leaves everything else untouched.",
        {
            "path": {"type": "string", "description": "File name, e.g. 'greet.py'."},
            "old_str": {
                "type": "string",
                "description": "The exact text to find, copied from view_file without the line numbers.",
            },
            "new_str": {"type": "string", "description": "The text to put in its place."},
        },
        ["path", "old_str", "new_str"],
    ),
    _schema(
        "insert_line",
        "Insert a new line of text after a given line number. Use 0 to insert "
        "at the very top. Use this when adding something rather than changing "
        "existing text, such as a new import.",
        {
            "path": {"type": "string", "description": "File name, e.g. 'greet.py'."},
            "line_number": {
                "type": "number",
                "description": "Insert after this line. 0 means the top of the file.",
            },
            "text": {"type": "string", "description": "The line to insert, without a newline."},
        },
        ["path", "line_number", "text"],
    ),
    _schema(
        "undo_edit",
        "Undo the last change made to a file, restoring its previous contents. "
        "Use this if an edit turned out wrong.",
        {"path": {"type": "string", "description": "File name, e.g. 'greet.py'."}},
        ["path"],
    ),
]

TOOL_FUNCTIONS = {
    "list_files": list_files,
    "view_file": view_file,
    "create_file": create_file,
    "str_replace": str_replace,
    "insert_line": insert_line,
    "undo_edit": undo_edit,
}


# --------------------------------------------------------------------------
# The conversation loop - same shape as multi_turn_conversion_with_tools.py
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful file editing assistant working in a small workspace. "
    "Before editing an existing file, always view it first so the text you "
    "quote matches exactly. Prefer str_replace over rewriting a whole file. "
    "When you are done, say briefly what you changed."
)


def chat_with_tools(messages, temperature=0.0):
    options = {"messages": messages, "temperature": temperature, "tools": TOOLS}
    try:
        return client.chat.completions.create(model=MODEL, **options).choices[0].message
    except RateLimitError:
        print("(OpenRouter rate limit hit, falling back to Groq...)")
        return (
            groq_client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=4096,
                extra_body={"reasoning_effort": "low"},
                **options,
            )
            .choices[0]
            .message
        )


def run_tool(tool_call):
    name = tool_call.function.name
    if name not in TOOL_FUNCTIONS:
        return f"Error: no tool named {name}."

    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return f"Error: could not parse arguments for {name}."

    print(f"[{name} {arguments}]")
    try:
        return str(TOOL_FUNCTIONS[name](**arguments))
    except Exception as error:  # pylint: disable=broad-except
        # Errors go back to the model as text. That matters more here than for
        # read-only tools: "old_str appears 3 times" is an instruction the
        # model can act on by quoting more context and trying again.
        return f"Error running {name}: {error}"


def send(messages, user_text, max_turns=8):
    """Add a message to an ongoing conversation and run it to an answer.

    Takes the messages list rather than owning it, so the same conversation can
    continue across several requests. That is what makes this multi-turn: the
    model can refer to a file it created three messages ago.
    """
    messages.append({"role": "user", "content": user_text})

    for _ in range(max_turns):
        message = chat_with_tools(messages)

        entry = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        messages.append(entry)

        if not message.tool_calls:
            return message.content

        for call in message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": run_tool(call),
                }
            )

    return "Gave up: too many tool calls without an answer."


if __name__ == "__main__":
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    REQUESTS = [
        # Write a file from nothing.
        "Create a file called greet.py with a function greet(name) that prints "
        "a greeting, and a call to greet('World').",
        # Edit it. The model has to view the file first to quote the old text
        # exactly, so this takes several tool calls.
        "Change the greeting so it says 'Hey' instead of 'Hello', and make it "
        "greet 'Kamran' instead of 'World'.",
        # Reads back the file it edited earlier in the same conversation.
        "Show me the final contents of greet.py.",
    ]

    for request in REQUESTS:
        print(f"\n=== {request}")
        print(send(conversation, request))

    print(f"\nFiles now in {SANDBOX.name}: {list_files()}")
