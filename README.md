# ProjektBadawczyHOI
1. Pobierz https://github.com/vt-vl-lab/DRG
2. Zmień ścieżkę w pliku: /my_tools/my_configs/RCNN.yaml
3. W celu stworzenia własnych plików pickle: należy uruchomić create_pickles.py.
( W celu uruchomienia kolejnych plików nie jest konieczne, w repo są juz utworzone przykladowe pliki).
4. Plik: count_AP.py - tworzy plik .csv z wynikami (transformation_level, agent_mAP, agent_TP,agent_FN,agent_FP,role_mAP,role_TP,role_FN,role_FP) w lokalizacji /my_tools/test_files/...
5. Plik: predict_one_image.py - predykcja dla zdjec testowych:
Dla podanego pliku .jpg wygeneruje plik .txt z zawartością typu "ski(41.80%) using skis(68.59%)".
