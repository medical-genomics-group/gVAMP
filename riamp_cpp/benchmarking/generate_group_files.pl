#!/usr/bin/perl

use warnings;
use strict;

my $NGROUPS = 1;
my $M = 10000;

# Group index file
open F, ">test1.gri" or die $!;
open G, ">test1.gri_hydra" or die $!;
for my $i (0..($M-1)) {
    my $mgroup = rand($NGROUPS);
    printf F ("%d %d\n", $i, $mgroup);
    printf G ("%d\n", $mgroup);
}
close F;
close G;

# Group mixture file (single line)
open F, ">test1.grm" or die $!;
open G, ">test1.grm_hydra" or die $!;
for my $i (1..$NGROUPS) {
    printf F ("%.5f %.5f %.5f %.5f\n", 0.0, 0.0001 * $i, 0.001 * $i, 0.01 * $i);
    printf G ("%.5f,%.5f,%.5f;", 0.0001 * $i, 0.001 * $i, 0.01 * $i);
}
close F;
close G;

