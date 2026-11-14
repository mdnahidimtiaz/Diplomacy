from config import TOPICS, REGIONS


def extract_answer_lines(text):
    lines = []
    for line in text.split("\n"):
        line = line.strip().lower().lstrip("*#- ").replace("**", "")
        if line.startswith(("topics:", "regions:")):
            lines.append(line)
    return "\n".join(lines)


def merge_chunk_outputs(chunk_answers, valid_topics=TOPICS, valid_regions=REGIONS):
    topics, regions = [], []
    for answer in chunk_answers:
        for line in answer.split("\n"):
            if line.startswith("topics:"):
                items = [t.strip() for t in line.replace("topics:", "").split(",")]
                topics += [t for t in items if t in valid_topics and t not in topics]
            elif line.startswith("regions:"):
                items = [r.strip() for r in line.replace("regions:", "").split(",")]
                regions += [r for r in items if r in valid_regions and r not in regions]
    merged = f"topics: {', '.join(topics)}\nregions: {', '.join(regions)}"
    return merged, topics, regions
