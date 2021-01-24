# -*- coding: utf-8 -*-
# Copyright(c) 2009 - present CNRS
# All rights reserved.



class ChineseDepGraph:

    ROOT_TOKEN = '<root>'

    def __init__(self,sent_id,deprel,feats,edges,wordlist=None,upos_tags=None,xpos_tags=None,with_root=False):

        self.gov2dep = { }
        self.has_gov = set()            #set of nodes with a governor
        self.root_id = None
        self.sentence_id = sent_id


        for (gov,label,dep) in edges:
            self.add_arc(gov,label,dep)

        if with_root:
            self.add_root()

        if wordlist is None:
            wordlist = []
        self.words    = [DepGraph.ROOT_TOKEN] + wordlist
        self.upos_tags = [DepGraph.ROOT_TOKEN] + upos_tags if upos_tags else None
        self.xpos_tags = [DepGraph.ROOT_TOKEN] + xpos_tags if xpos_tags else None
        self.feats_aspect    = [DepGraph.ROOT_TOKEN] + feats
        self.deprel = [DepGraph.ROOT_TOKEN] + deprel
   def add_root(self):

        if self.gov2dep and 0 not in self.gov2dep:
            root = list(set(self.gov2dep) - self.has_gov)
            if len(root) == 1:
                self.add_arc(0,'root',root[0])
                self.root_id = root[0]
            else:
                assert(False) #no single root... problem.
        elif not self.gov2dep: #single word sentence
            self.add_arc(0,'root',1)

    def add_arc(self,gov,label,dep):
        """
        Adds an arc to the dep graph
        """
        if gov in self.gov2dep:
            self.gov2dep[gov].append( (gov,label,dep) )
        else:
            self.gov2dep[gov] = [(gov,label,dep)]

        self.has_gov.add(dep)

    @staticmethod
    def read_tree(istream):
        """
        Reads a conll tree from input stream
        """
        def graph(conll,sent_id):
            words   = [ ]
            upostags = [ ]
            xpostags = [ ]
            edges   = [ ]
            feats_aspect   = [ ]
            deprel = [ ]
            for dataline in conll:
                if '-' in dataline[0]:
                    continue          #skips compound word annotation
                words.append(dataline[1])
                upostags.append(dataline[3])
                xpostags.append(dataline[4])
                deprel.append(dataline[7].strip())
                if dataline[6] != '0': #do not add root immediately
                    edges.append((int(dataline[6]),dataline[7].strip(),int(dataline[0]))) # shift indexes !

                if dataline[5] != '_' and 'Aspect' in dataline[5]: # éliminer les verbes participe passé
                    aspect= dataline[5].split('=')[1]
                    #feat_tense = [x for x in feats_list if 'Tense' in x]
                    feats_aspect.append(aspect)
                else:
                    feats_aspect.append('_')
            return DepGraph(sent_id,deprel,feats_aspect,edges,words,upos_tags=upostags,xpos_tags=xpostags,with_root=True)

        conll = []
        deptrees = []
        line = istream.readline()  # a string
        # checks whether the string consists of whitespace or contains #
        while istream and line.startswith('#'):  # (line.isspace() or ):
            if line.startswith('# sentenceID'):
                sent_id = line.split("=")[1].strip()
                sent_id = sent_id.split('_')
                # print("#1",sent_id)
                if len(sent_id[1]) == 1:
                    sent_id = int(sent_id[0] + '0' + sent_id[1])
                else:
                    sent_id = int(sent_id[0] + sent_id[1])
            line = istream.readline()

        while istream and not line.strip() == '':
            line = line.split('\t')  # split this String by tabulator into an array
            conll.append(line)
            line = istream.readline()
            while line.startswith('#'):
                if line.startswith('# sentenceID'):
                    deptree = graph(conll, sent_id)
                    deptrees.append(deptree)
                    conll = []
                    sent_id = line.split("=")[1].strip()
                    sent_id = sent_id.split('_')
                    # print("#2",sent_id)
                    if len(sent_id[1]) == 1:
                        sent_id = int(sent_id[0] + '0' + sent_id[1])
                    else:
                        sent_id = int(sent_id[0] + sent_id[1])
                line = istream.readline()

        if not conll:
            return None
        deptrees.append(graph(conll, sent_id))
        return deptrees

    def __str__(self):
        """
        Conll string for the dep tree
        """
        lines    = [ ]
        revdeps  = dict([( dep, (label,gov) ) for node in self.gov2dep for (gov,label,dep) in self.gov2dep[node] ])
        for node in range( 1, len(self.words)  ):
            L    = ['-']*11
            L[0] = str(node)
            L[1] = self.words[node]
            if self.upos_tags:
                L[3] = self.upos_tags[node]
            if self.xpos_tags:
                L[4] = self.xpos_tags[node]
            label,head = revdeps[node] if node in revdeps else ('root', 0)
            if self.feats_aspect:
                L[5] = self.feats_aspect[node]
            if self.deprel:
                L[7] = self.deprel[node]
            L[6] = str(head)
            lines.append( '\t'.join(L))
        return '\n'.join(lines)

    def __len__(self):
        return len(self.words)

def treebank(filename):
    istream = open(filename)
    tlist = []
    deptrees = ChineseDepGraph.read_tree(istream)
    while deptrees :
        tlist.append(deptrees)
        deptrees = ChineseDepGraph.read_tree(istream)
    istream.close()
    return tlist


