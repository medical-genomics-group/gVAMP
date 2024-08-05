## Running gVAMP on DNAnexus platform 


An example of running gVAMP using cloud workstation on DNAnexus platform.
1. Prepare folder structure (modify `run_gvamp.sh` if using different naming)
```
mkdir mnt
mkdir mnt/output
```

2.  Download PLINK format bed, bim and phen file into workstation
```
cd mnt
#dx download ... 
cd ../
```

3. Clone gVAMP repository
```
git clone https://github.com/medical-genomics-group/gVAMP.git
```

4. Compile gVAMP executable
```
cd gVAMP
mpic++ main_real.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o main_real.exe
cd ../
```

5. Copy `run_gvamp.sh` script and modify it, specifying the paths and input parameters
```
cp gVAMP/dnanexus_example/run_gvamp.sh .
```

6. Run gVAMP in the background
```
bash run_gvamp.sh &
```