from pathlib import Path

def main():
    # inpout file data\records\Gragirene\Geisinger CMC\Thornhill, Janice - Geisinger.txt
    BASE_DIR = Path(__file__).resolve().parent
    print(f"Base Directory: {BASE_DIR}")
    # C:\Users\esupp\Documents\Projects\python\ai\process-medical-records\output\Penley\Penley - York Hospital T3 Ortho-Neuro.txt
    filename = BASE_DIR / "output" / "Penley" / "Wellspan Home Care" / "Penley - WellSpan VNA Home Care-cleaned-final.txt" 
    # C:\Users\esupp\Documents\Projects\python\ai\process-medical-records\output\Penley\Wellspan Home Care\Penley - WellSpan VNA Home Care-cleaned-final.txt
    # C:\Users\esupp\Documents\Projects\python\ai\process-medical-records\output\Penley - WellSpan VNA Home Care.txt
    # count the number of words in a text file
    
    with open(filename, 'r', encoding="utf-8") as file:
        text = file.read()
        words = text.split()
        word_count = len(words)
        token_count = word_count * 1.4
        print(f"Number of words in {filename.stem}: {word_count}")
        print(f"Number of tokens in {filename.stem}: {token_count}")
    
if __name__ == "__main__":
    main()