import os
import json
from collections import Counter
from nltk.tokenize import sent_tokenize
from operator import itemgetter
import codecs

class TextSumFormatter:
    def fileCount(self):
        return sum([len(files) for r,d,files in os.walk(self.fromPath)])

    def processPath(self, toPath):
        try:
            fout = open(os.path.join(toPath, '{}-{}'.format(self.baseName, self.fileNdx)), 'a+')
            for path, dirs, files in os.walk(self.fromPath):
                for fn in files:
                    fullpath = os.path.join(path, fn)
                    if os.path.isfile(fullpath):
                        
                        #with open(fullpath, "rb") as f:
                        with codecs.open(fullpath, "rb", 'ascii', "ignore") as f:
                            try:
                                textSumFmt = self.textsumFmt
                                finalRes = textSumFmt["artPref"]
                                content = f.readlines()
                                self.populateVocab(content)

                                sentences = sent_tokenize((content[1]).encode('ascii', "ignore").strip('\n'))
                                for sent in sentences:
                                    finalRes += textSumFmt["sentPref"] + sent.replace("=", "equals") + textSumFmt["sentPost"] 
                                finalRes += ((textSumFmt["postVal"] + '\t' + textSumFmt["absPref"] + textSumFmt["sentPref"] + (content[0]).strip('\n').replace("=", "equals") + textSumFmt["sentPost"] + textSumFmt["postVal"]) + '\t' +'publisher=' + self.publisher + os.linesep)
                                
                                if self.lineNdx != 0 and self.lineNdx % self.lines == 0:
                                    fout.close()
                                    self.fileNdx+=1
                                    fout = open(os.path.join(toPath, '{}-{}'.format(self.baseName, self.fileNdx)), 'a+')

                                fout.write( ("{}").format( finalRes.encode('utf-8', "ignore") ) )
                                self.lineNdx+=1
                            except RuntimeError as e:
                                print "Runtime Error: {0} : {1}".format(e.errno, e.strerror)
        finally:
            fout.close()
    
    def populateVocab(self, content):
        title = (content[0]).strip('\n')
        body = (content[1]).strip('\n')
        wordCnt = Counter(title.split())
        for item in wordCnt.items():
            self.incJsonElement(item[0], item[1])
            #print item

    def incJsonElement(self, passedKey = "", cnt=0):
        if passedKey == "":
            return
        if passedKey not in self.vocab:
            #print "Not here"
            self.vocab[passedKey] = cnt
        else:
            #print "Here"
            self.vocab[passedKey] += cnt
    
    def outputVocab(self, path):
        #First perform sort and then write out
        vocabSorted = sorted(self.vocab.items(), key=itemgetter(1), reverse=True)
        with open(path,'w+') as v:
            for item in vocabSorted:
                print >>v, ("{} {}".format(item[0], str(item[1])) )

    def toTrainPath(self):
        return self.toPathTrain
    
    def toTestPath(self):
        return self.toPathTest
    
    def lineNdx(self):
        return self.lineNdx
    
    def totTrainCnt(self):
        return self.totTrainCnt
    
    def vocab(self):
        return self.vocab

    def __init__(self, fromPath, toPathTrain, toPathTest, publisher, baseName, fileNdx, lineNdx, lines,textSumFmt):
        self.fromPath = fromPath
        self.toPathTrain = toPathTrain
        self.toPathTest = toPathTest
        self.publisher = publisher
        self.totFileCnt = self.fileCount()
        self.totTrainCnt = (int(self.totFileCnt * 0.8))
        self.baseName = baseName
        self.fileNdx = fileNdx
        self.lineNdx = lineNdx
        self.lines = lines
        self.textsumFmt = textSumFmt
        self.vocab = {"<UNK>":1, "<s>":1, "</s>":1, "<PAD>":1,"<d>":1,"</d>":1,"<p>":1,"</p>":1}

def main():
    buzzTextSumFormatter = TextSumFormatter("./articlesForTraining",
                                            "./srcArticlesTrain",
                                            "./srcArticlesTest",
                                            "REUTERS",
                                            #"./tempArt",
                                            #"./tempArtTrain",
                                            #"./tempArtTest",
                                            #"TEMP",
                                            "data",
                                            0,
                                            0,
                                            100,
                                            json.loads('{ "artPref":"article= <d> <p> ","absPref":"abstract= <d> <p> ","postVal":" </p> </d> ","sentPref":"<s> ", "sentPost":" </s> "}'))

    try:
        totalTrainCount = buzzTextSumFormatter.totTrainCnt
        while (buzzTextSumFormatter.lineNdx < totalTrainCount):
            buzzTextSumFormatter.processPath(buzzTextSumFormatter.toTrainPath())
        #end while
        print "We are out"
        buzzTextSumFormatter.outputVocab("vocab")
        buzzTextSumFormatter.processPath(buzzTextSumFormatter.toTestPath())
        print "All processing complete"

    except RuntimeError as e:
        print "Runtime Error: {0} : {1}".format(e.errno, e.strerror)



if __name__ == '__main__':
    main()  
