#!/usr/bin/perl
use warnings;
use strict;

# Info on dataset to process
# --------------------------
#my $DATADIR = "$ENV{HOME}/CADMOS/Matthew/BayesRRcmd/test/data/testdata_msp_constpop_Ne10K_M100K_N10K";
#my $DATASET = "testdata_msp_constpop_Ne10K_M100K_N10K";

my $DATADIR = "/scratch/orliac/testN500K";
my $DATASET = "testN500K";

my $M       = 114560;      # Number of markers
$M = 2000;

my $N       = 10000;       # Number of individuals
$N = 500000;

my $CL      = 5;        # Number of iterations (chain length)
#my $SEED    = 1;           # Seed for RNG
my $SM      = 1;           # Marker shuffling switch
my $EXE     = "hydra"; # Binary to run

die unless -d $DATADIR;
die unless (-f $EXE && -e $EXE);

# Benchmark plan & processing setup
# ---------------------------------
my @NNODES    = qw(1 2 4 8 16);
@NNODES    = qw(2);
my @NTPN      = qw(1);
my @SYNCR     = qw(1 2 4 8 16 32 64 128);
@SYNCR     = qw(1);
my @SEEDS = qw(0 10 20 30 40 50 60 70 80 90);
@SEEDS = qw(1);

my $PARTITION = 'parallel';
#$PARTITION = 'debug';

my $mem_tot = ($M * $N * 8 + $M * $N / 4) * 1E-9;
printf("Total RAM needed to store Cx in raw and DP: %.1f GB.\n", $mem_tot);

my $nickname = "bench_${EXE}";

my $submit = "submit_all_sbatch_$nickname.sh";
open S, ">$submit" or die $!;

foreach my $nnodes (@NNODES) {

    foreach my $ntpn (@NTPN) {

        my $ntasks = $nnodes * $ntpn;
        printf("Total number of tasks: %d (%d task(s) on %d node(s))\n", $ntasks, $ntpn, $nnodes);

        # Assuming 1 CPU per task
        my $cpu_per_task = 1;
        my $mem_per_node = int(($mem_tot / ($nnodes) + 0.5) * 3);
        printf("Memory per node: %d GB.\n", $mem_per_node);

        foreach my $syncr (@SYNCR) {

            foreach my $SEED (@SEEDS) {

                #printf("nodes: $nnodes, tasks per node: $ntpn, syncr: $syncr\n");
                my $basename = sprintf("${nickname}__nodes_%02d__tpn_${ntpn}__tasks_%02d__cl_${CL}__syncr_%03d__seed_%02d", $nnodes, $ntasks, $syncr, $SEED);
                
                # Delete output file if already existing
                my $csv = $basename.'.csv';
                if (-f $csv) {
                    unlink $csv;
                    print "INFO: deleted file $csv\n";
                }

                open F, ">$basename.sh" or die $!;

                print F "#!/bin/bash\n\n";
                #print F "#SBATCH --nodes $nnodes\n";
                #print F "#SBATCH --exclusive\n";
                #print F "#SBATCH --mem ${mem_per_node}G\n";
                #print F "#SBATCh --ntasks $ntasks\n";
                #print F "#SBATCH --ntasks-per-node $ntpn\n";
                #print F "#SBATCH --cpus-per-task 1\n";
                #print F "#SBATCH --time 02:00:00\n";
                #print F "#SBATCH --partition $PARTITION\n";
                #print F "#SBATCH --constraint=E5v4\n";
                #print F "#SBATCH --output ${basename}__jobid\%J.out\n";
                #print F "#SBATCH --error  ${basename}__jobid\%J.err\n";
                print F "\n";
                print F "module load intel intel-mpi eigen boost\n\n";
                print F "env | grep SLURM\n\n";
                print F "\n";
                print F "start_time=\"\$(date -u +\%s)\"\n";
                print F "\n";
                print F "srun --ntasks-per-node $ntpn --cpus-per-task 1 --time 02:00:00 --output ${basename}__jobid\%J.out --error  ${basename}__jobid\%J.err $EXE --bfile $DATADIR/$DATASET --pheno $DATADIR/$DATASET.phen --chain-length $CL --seed $SEED --shuf-mark $SM --mpi-sync-rate $syncr --number-markers $M --mcmc-samples ${basename}.csv\n";
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
