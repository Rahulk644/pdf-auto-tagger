import tagger.pipeline as pipe

file_path = pipe.__file__
with open(file_path, "r") as f:
    content = f.read()

content = content.replace("                    metadata=el.metadata,\n", "")

with open(file_path, "w") as f:
    f.write(content)
