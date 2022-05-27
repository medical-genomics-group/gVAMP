#!/usr/bin/perl

use warnings;
use strict;
use File::Path qw(make_path remove_tree);

# Info on dataset to process
# --------------------------
my ($datadir, $dataset, $phen, $S, $sparsedir, $sparsebsn, $numinds, $numsnps) = ("", "", "", "0.1,1.0");

# Select dataset here
my $DS = 3;
my $frombed=0;

if ($DS == 0) {
    $datadir="./test/data";
    $dataset="uk10k_chr1_1mb";
    $phen="test";
    $sparsedir=$datadir;
    $sparsebsn=${dataset}."_uint";
    $numinds=3642;
    $numsnps=6717;
} elsif ($DS == 1) {
    $datadir="/scratch/orliac/testM100K_N5K_missing";
    $dataset="memtest_M100K_N5K_missing0.01";
    $phen="memtest_M100K_N5K_missing0.01";
    $sparsedir=$datadir;
    $sparsebsn=${dataset}."_uint";
    $numinds=5000;
    $numsnps=117148;
} elsif ($DS == 2) {
    $datadir="/scratch/orliac/testN500K";
    $dataset="testN500K";
    $phen=$dataset;
    $sparsedir=$datadir;
    $sparsebsn=${dataset}."_uint";
} elsif ($DS == 3) {
    $sparsedir="/scratch/orliac/UKBgen/";
    $sparsebsn="epfl_test_data_sparse";
    $phen="epfl_test_data";
    $numinds=457810;
    $numsnps=8430446;
    $numsnps=100000;
    $S="0.00001,0.0001,0.001,0.01"
} else {
    die "Unknown dataset selected: $DS!";
}

my $dir = $frombed == 1 ? $datadir : $sparsedir;
my $set = $frombed == 1 ? $dataset : $sparsebsn;

my $COV       = "--covariates $dir/scaled_covariates.csv"; 
$COV          = "";
my $BLK       = "--marker-blocks-file $dir/${dataset}.blk";
$BLK          = "";

my $CL        = 2;        # Number of iterations (chain length)
my $THIN      = 10;
my $SAVE      = 10;
my $SM        = 1;          # Marker shuffling switch
my $MEMGB     = 180;        # Helvetios
my $EXE       = "/home/orliac/DCSR/CTGG/BayesRRcmd/src/hydra";

die unless -d $dir;
die unless (-f $EXE && -e $EXE);

# Benchmark plan & processing setup
# ---------------------------------
my $nickname = "UKBgenV01";

my $DIR    = "/scratch/orliac/UKBgen/runs/$nickname";
unless (-d $DIR) { make_path($DIR) or die "mkdir($DIR): $!"; }
my @NNODES = qw(1);
my @NTPN   = qw(100);
my @SYNCR  = qw(5);
my @SEEDS  = qw(4321);

my $PARTITION = 'parallel';
#$PARTITION = 'debug';

my $submit = "$DIR/submit_all_sbatch_$nickname.sh";
open S, ">$submit" or die $!;

foreach my $nnodes (@NNODES) {

    foreach my $ntpn (@NTPN) {

        my $ntasks = $nnodes * $ntpn;
        printf("Total number of tasks: %4d ( = %3d x %3d)\n", $ntasks, $ntpn, $nnodes);

        # Assuming 1 CPU per task
        my $cpu_per_task = 1;
        my $mem_per_node = $MEMGB;

        foreach my $syncr (@SYNCR) {

            foreach my $SEED (@SEEDS) {

                #printf("nodes: $nnodes, tasks per node: $ntpn, syncr: $syncr\n");
                my $basename = sprintf("${nickname}__nodes_%02d__tpn_%02d__tasks_%02d__cl_${CL}__syncr_%03d__seed_%02d", $nnodes, $ntpn, $ntasks, $syncr, $SEED);
                
                open F, ">$DIR/$basename.sh" or die $!;

                print F "#!/bin/bash\n\n";
                print F "#SBATCH --account ext-unil-ctgg\n";
                #print F "#SBATCH --nodes $nnodes\n";
                print F "#SBATCH --exclusive\n";
                print F "#SBATCH --mem ${mem_per_node}G\n";
                print F "#SBATCH --ntasks $ntasks\n";
                #print F "#SBATCH --ntasks-per-node $ntpn\n";
                print F "#SBATCH --cpus-per-task 1\n";
                print F "#SBATCH --time 0-00:30:00\n";
                print F "#SBATCH --partition $PARTITION\n";
                #print F "#SBATCH --constraint=E5v4\n";
                print F "#SBATCH --output ${basename}__jobid\%J.out\n";
                print F "#SBATCH --error  ${basename}__jobid\%J.err\n";
                print F "\n";
                print F "module load intel intel-mpi eigen boost\n\n";
                print F "env | grep SLURM\n\n";
                print F "\n";
                print F "start_time=\"\$(date -u +\%s)\"\n";
                print F "\n";
                if ($frombed == 0) {
                    print F "srun $EXE --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out $basename --seed $SEED --shuf-mark $SM --mpi-sync-rate $syncr --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $COV $BLK  --number-individuals $numinds --number-markers $numsnps || exit 1\n";
                } else { 
                    die "Adapt for from bed!";
                }
                print F "\n";
                print F "end_time=\"\$(date -u +\%s)\"\n";
                print F "elapsed=\"\$((\$end_time-\$start_time))\"\n";
                print F "echo \"Total time in sec: \$elapsed\"\n";
                close F;
                print S "sbatch $basename.sh\n";
            }
        }
    }
}

close S;

print "\nWrote $submit. To submit: sh $submit\n\n";
