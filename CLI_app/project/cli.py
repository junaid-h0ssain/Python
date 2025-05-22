import typer
import json
import os

db = "db.json"
app = typer.Typer()

@app.command()
def add(tn: str,tp: str):
    tid = 0
    tasks=[]
    if os.path.exists(db):
        with open(db,"r") as f:
            tasks=json.load(f)
            for task in tasks:
                id = task['task_id']
                tid =id
                #print(id)
    tid +=1
    task = {
        'task_id':tid,
        "task_name":tn,
        "task_progress":tp
    }
    
    
    tasks.append(task)

    with open(db, "w") as f:
        json.dump(tasks, f,indent=4)

    print(f"Task '{task}' saved to {db}.")


@app.command()
def list():
    with open(db, "r") as f:
        tasks = json.load(f)

    typer.secho("\nto-do list:\n", fg=typer.colors.BLUE, bold=True) 
    columns=(
        "ID. ",
        "| Name    ",
        "| Progress    ",
    )
    headers = "".join(columns)
    typer.secho(headers, fg=typer.colors.BLUE, bold=True)
    typer.secho("-" * len(headers), fg=typer.colors.BLUE, bold=True)

    for task in tasks:
        typer.secho(
            f"{task['task_id']}{(len(columns[0]) - len(str(tasks.index(task)))-1) * ' '}"
            f" | {task['task_name']}{(len(columns[1]) - len(str(task['task_name']))-3) * ' '}"
            f" | {task['task_progress']}",#"{(len(columns[2]) - len(str(task['task_progress']))) * ' '}",
            fg=typer.colors.GREEN,
            bold=True,
        )
    typer.secho("-" * len(headers), fg=typer.colors.BRIGHT_RED, bold=True)

@app.command()
def update(tid: int,tn: str,tp : str):
    tasks=[]
    if os.path.exists(db):
        with open(db,"r") as f:
            tasks=json.load(f)

    for task in tasks:
        if tid == task['task_id']:
            tasks.remove(task)
            task = {
                'task_id':tid,
                "task_name":tn,
                "task_progress":tp
            }
            tasks.append(task)
            print(f"Task '{task}' updated to {db}.")
            break

    with open(db, "w") as f:
        json.dump(tasks, f,indent=4)


@app.command()
def delete(tid: int):
    tasks=[]
    if os.path.exists(db):
        with open(db,"r") as f:
            tasks=json.load(f)

    for task in tasks:
        if tid == task['task_id']:
            tasks.remove(task)
            print(f"Task '{task}' deleted from {db}.")
            break

    with open(db, "w") as f:
        json.dump(tasks, f,indent=4)


if __name__ == "__main__":
    app()
    id = 0