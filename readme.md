```sh
#* cpu
python run.py  --opt -2 --size 500 --verbose 0 --pivoting 0 --check_equal 1
python run.py  --opt -2 --size 1200 --verbose 0 --pivoting 0 --check_equal 0

#* gpu
python run.py  --opt 2 --size  500 --verbose 0 --pivoting 1 --check_equal 1
python run.py  --opt 2 --size 1200 --verbose 0 --pivoting 1 --check_equal 0
```

- time: ms  

|n    |cpu  |block LU (cuda)|
|---- |-----|---------------|
|200  |    7|     25        |
|300  |   25|     31        |
|400  |   59|     34        |
|500  |  121|     39        |
|600  |  201|     41        |
|1000 |  942|     59        |
|1200 | 1618|     68        |
|1300 | 2124|     72        |
|1600 | 3997|     88        |
|2000 | 7990|    104        |
|4000 |70105|    282        |
|8000 |     |   1025        |
|16000|     |   4469        |
|20000|     |   7388        |
|25000|     |  12421        |
|30000|     |  19289        |
|31000|     |  20913        |