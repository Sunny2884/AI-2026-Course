"""Convert a user-prepared JSONL to portable-source absolute training pairs."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--data-root',type=Path,required=True);p.add_argument('--target-root',type=Path,required=True)
a=p.parse_args();rows=[]
for line in a.input.read_text().splitlines():
 if not line.strip():continue
 r=json.loads(line)
 source=Path(r.get('source_path',r.get('image_path','')));target=Path(r['target_path'])
 if not source.is_absolute():source=a.data_root/source
 if not target.is_absolute():target=a.target_root/target
 if not source.is_file() or not target.is_file():raise FileNotFoundError((source,target))
 rows.append(dict(id=r.get('id',source.stem),source_path=str(source.resolve()),target_path=str(target.resolve())))
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(''.join(json.dumps(r)+'\n' for r in rows));a.output.chmod(0o777)
