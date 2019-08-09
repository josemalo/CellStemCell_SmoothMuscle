#!/usr/bin/env python
# date: 2019-01-29

"""
scdiffAltRun.py
"""

from __future__ import print_function

___author___ = 'Jose Malagon-Lopez'

import scdiff as S
import argparse
import os


"""
Run MODIFIED scdiff to get a single-cell trajectory inference where descendant cells are similar to their parents in terms of 
gene expression levels and their regulatory information.
"""
def scdiffRun(mainInput, TF, clusterInfo, clusterID, outName, outputDir):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    print('Reading Input')
    # read main input
    tabData = S.TabFile(mainInput).read('\t')
    # data table without header
    mainTable = tabData[1:]
    # gene names
    geneList = tabData[0][3:]

    # collect cells
    allCells = []

    print('Generating Cell Instance')
    for line in mainTable:
        # input: Cell_ID, TimePoint, Expression, typeLabel, GeneList
        expTab = [float(item) for item in line[3:]]
        allCells.append(S.Cell(line[0], float(line[1]), expTab, line[2], geneList))

    print('Adding VAE cluster labels to every cell')
    vaeLabels = []
    with open(clusterID, 'r') as f:
        allLines = f.readlines()
        for label in allLines:
            vaeLabels.append(int(label.strip()))

    for i in range(len(allCells)):
        allCells[i].Label = vaeLabels[i]

    print('Creating virtual ancestor with its "cluster" label and added to the cells')
    # create virtual ancestor with its 'cluster' label and added to the cells
    va = S.buildVirtualAncestor(allCells)
    va.Label = 0
    allCells.insert(0,va)

    ### Obtain the Cell-sets trajectories as a Graph object
    # S.Graph inputs: Cells, tfdna, kc, largeType=None, dsync=None, virtualAncestor=None, fChangeCut=1, etfile=None
    mainGraph = S.Graph(allCells, TF, clusterInfo, virtualAncestor="1", dsync="1")

    # attributes
    gNodes = mainGraph.Nodes
    gEdges = mainGraph.Edges
    # gdtd, gdtg, gdmb = mainGraph.dTD, mainGraph.dTG, mainGraph.dMb

    # collect Nodes information
    print("Collecting Nodes Data")
    out_nodes = os.path.join(outputDir, "_".join([outName, 'Nodes.txt']))

    with open(out_nodes, 'w') as out_table:
        print('ID','Index', 'TimePoint', 'Time', 'TotalCells', 'ParentID', 'ParentIndex', 'TotalChildren', 'ChildrenIndexes', sep='\t', file=out_table)

        for node in gNodes:
            if node.P is None:
                parentid, parentindex = '.', '.'
            else:
                parentid, parentindex = str(node.P.ID), str(gNodes.index(node.P))
            if node.C is None:
                childrenindex = '.'
            else:
                childrenindex = '_'.join([str(gNodes.index(x)) for x in node.C])

            outline=[node.ID,str(gNodes.index(node)),str(node.T),str(node.ST),str(len(node.cells)),parentid,parentindex,str(len(node.C)),childrenindex]
            print(*outline, sep='\t', file=out_table)

    # collect Cells information
    print("Collecting Cells Data")
    out_cells = os.path.join(outputDir, "_".join([outName, 'Cells.txt']))

    with open(out_cells, 'w') as out_table:
        print('cell.ID', 'cell.StoredTime' ,'cell.Label' ,'cell.TypeLabel' ,'Node.ID','Node.Index', 'Node.TimePoint', 'Node.Time', sep='\t', file=out_table)

        for node in gNodes:
            for cell in node.cells:
                outline = [cell.ID, cell.T, cell.Label, cell.typeLabel, node.ID, str(gNodes.index(node)), str(node.T), str(node.ST)]
                print(*outline, sep='\t', file=out_table)

    # collect Paths information
    print("Collecting Paths Data")
    out_paths = os.path.join(outputDir, "_".join([outName, 'Paths.txt']))

    with open(out_paths, 'w') as out_table:
        print('path.index', 'nodeFrom.ID', 'nodeFrom.TimePoint', 'nodeFrom.Time', 'nodeTo.ID', 'nodeTo.TimePoint', 'nodeTo.Time', sep='\t', file=out_table)

        for path in gEdges:
            outline = [gEdges.index(path), path.fromNode.ID, path.fromNode.T, path.fromNode.ST, path.toNode.ID, path.toNode.T, path.toNode.ST]
            print(*outline, sep='\t', file=out_table)

def main():
    parser = argparse.ArgumentParser(description='Run scdiff and get the tables with the nodes and cells allocation in the differentition trajectory.')
    parser.add_argument("--mainInput", help="Tab separated input matrix with header. Columns: cell,time,label,gene expression.", required=True)
    parser.add_argument("--TF", help="Full path to transcription factor input table.", required=True)
    parser.add_argument("--clusterInfo", help="Tab separated input matrix without header. Columns: time,number of clusters per time point", default=None)
    parser.add_argument("--clusterID", help="Single Column and No header txt file with vae cluster labels. Ordered as in mainInput", default=None)
    parser.add_argument("--outName", help="Output basename' ", required=True)
    parser.add_argument("--outputDir", help="Output directory", required=True)
    args = parser.parse_args()

    print(args)

    scdiffRun(args.mainInput, args.TF, args.clusterInfo, args.clusterID, args.outName, args.outputDir)

if __name__ == "__main__":

    main()
