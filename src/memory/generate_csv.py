from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

def main(dir:Path, output:Path):
    if type(dir) == str: dir = Path(dir)

    if not dir.exists():
        raise FileNotFoundError(f"{dir} does not exist")

    df = pd.DataFrame(columns=["path", "teclas", "fecha"])
    
    for subdir in dir.iterdir():
        for file in subdir.iterdir():
            if file.is_file():
                filename = file.name
                if "_l_" in filename: filename = filename.replace("_l_", "_")
                elif "_r_" in filename: filename = filename.replace("_r_", "_")
                elif filename.startswith("caps_lock"): filename = filename.replace("caps_lock", "caps")

                if subdir.name == "F" and filename[0] == "1":
                    teclas = "F"
                    fecha = filename
                else:
                    teclas, _, fecha = filename.split("_", maxsplit=2)
                
                if "_" in fecha: fecha = fecha.split("_", maxsplit=1)[0]
                if "." in fecha: fecha = fecha.split(".", maxsplit=1)[0]
                
                df = pd.concat([df, pd.DataFrame({"path": [file], "teclas": [teclas], "fecha": [fecha]})], ignore_index=True)
    df.index = pd.to_datetime(df["fecha"],
     format="%H-%M-%S")
    df.sort_index(inplace=True)
    df.to_csv(output, index=False)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-D", "--indir", help="input dir")
    parser.add_argument("-o", "--out", help="output file")
    args = parser.parse_args()

    main(args.indir, args.out)
