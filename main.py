import sys
from face_comparison import face_comparison
def main():
    if sys.argv[1]=='face_comparison':
        face_comparison()

if __name__ == '__main__':
    main()