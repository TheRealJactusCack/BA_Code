import time, datetime, pathlib

log = pathlib.Path(r"C:\temp\bg_test.log")
log.parent.mkdir(parents=True, exist_ok=True)

i = 0
while True:
    i += 1
    print(i)
    log.write_text(f"{datetime.datetime.now().isoformat()} tick {i}\n", encoding="utf-8", errors="ignore") if i == 1 else None
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} tick {i}\n")
    time.sleep(1)
