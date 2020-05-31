#!/bin/bash
rsync -tPvr --delete-excluded --delete --ignore-errors \
    --exclude '.git/*' \
    --exclude '.cache/' \
    --exclude '.netrwhist' \
    /home/zelhar/FU/bachelorarbeit/ ./ba/

#    --files-from='./dotfileslist.txt' \

