from pathlib import Path
import os
import shutil

path_videos = Path(__file__).parent / "videos"

def separar():
    for dir in path_videos.iterdir():
        nuevo_dir = "process_" + dir.name
        for file in dir.iterdir(): 
            if file.is_dir():
                continue

            if not file.name[0].isdigit() and (file.name[0] == file.name[0].upper()):
                print(file)
            else:
                Path(f"{path_videos}/{nuevo_dir}/").mkdir(exist_ok=True)

                shutil.move(f"{path_videos}/{dir.name}/{file.name}", f"{path_videos}/{nuevo_dir}/{file.name}")

def quitar_primera_letra():
    path_videos = Path(os.getcwd())
    print(path_videos)
    n = input("continue?")
    
    if n != "y":
        return

    for file in path_videos.iterdir():
        if file.is_dir():
            continue
        if file.name[0].isdigit():
            os.rename(f"{path_videos}/{file.name}", f"{path_videos}/{file.name[2:]}")
            
        # os.rename(f"{path_videos}/{file.name}", f"{path_videos}/{file.name[2:]}")

def resize_img():
    from PIL import Image
    import os

    path_videos = Path(os.getcwd())

    for file in path_videos.iterdir():
        if file.is_dir():
            continue
        
        img = Image.open(f"{path_videos}/{file.name}")
        img = img.resize((64, 36))
        img.save(f"{path_videos}/{file.name}")

def renombrar():
    path_videos = Path(os.getcwd())

    for file in path_videos.iterdir():
        if file.is_dir():
            continue
        
        pat = "A+W_0"
        if file.name.startswith(pat):
            name = file.name.replace(pat, "W_0")
            if Path(f"{path_videos}/{name}").exists():
                print(f"Existe: {path_videos}/{name}")
            else:
                os.rename(f"{path_videos}/{file.name}", f"{path_videos}/{name}")

def main():
    # separar()
    # quitar_primera_letra()
    # resize_img()
    renombrar()

if __name__ == "__main__":
    main()