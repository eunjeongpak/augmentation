# Augmentation
useful codes related to augmentation

### Last Updated : 2023.07.29

---

## aug.py

### 1. Data Structure

```buildoutcfg
[--example]
├─D
│ └─img_folder
│   └─all_ex
│        01na00ej000001kr.jpg
│        01na00ej000001kr.xml
```

### 2. CLI command

```buildoutcfg
[--h]
cd utils
python aug.py --dir {dir} --new_dir {new_dir} -- method {method}
[--example]
cd utils
python aug.py --dir 'D:/img_folder/all_ex/' --new_dir 'D:/img_folder/img_aug/' --method 'rddc'
```

- dir : original file path
- new_dir : the location where you want to save the results
- method : augmentation method
  - 'rddc': rotate & deep dark color
  - 'rdc': rotate & dark color
  - 'rldc': rotate & little dark color
  - 'rbc': rotate & bright color
  - 'rlc': rotate & light color
  - 'rn': rotate & noise
  - 'rgn': rotate & gaussian noise
  - 'fc': flip & color
  - 'ts': translation & shearing
  - 'crr': crop & resize & rotate

