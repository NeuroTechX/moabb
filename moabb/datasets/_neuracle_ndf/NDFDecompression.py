



class HuffNode(object):
    """
    定义一个HuffNode虚类，里面包含两个虚方法：
    1. 获取节点的权重函数
    2. 获取此节点是否是叶节点的函数

    """
    def get_wieght(self):
        raise NotImplementedError(
            "The Abstract Node Class doesn't define 'get_wieght'")

    def isleaf(self):
        raise NotImplementedError(
            "The Abstract Node Class doesn't define 'isleaf'")


class LeafNode(HuffNode):
    """
    树叶节点类
    """
    def __init__(self, value=0, freq=0,):
        """
        初始化 树节点 需要初始化的对象参数有 ：value及其出现的频率freq
        """
        super(LeafNode, self).__init__()
        # 节点的值
        self.value = value
        self.wieght = freq


    def isleaf(self):
        """
        基类的方法，返回True，代表是叶节点
        """
        return True

    def get_wieght(self):
        """
        基类的方法，返回对象属性 weight，表示对象的权重
        """
        return self.wieght

    def get_value(self):
        """
        获取叶子节点的 字符 的值
        """
        return self.value


class MiddleNode(HuffNode):
    """
    中间节点类
    """
    def __init__(self, left_child=None, right_child=None):
        """
        初始化 中间节点 需要初始化的对象参数有 ：left_child, right_chiled, weight
        """
        super(MiddleNode, self).__init__()

        # 节点的值
        self.wieght = left_child.get_wieght() + right_child.get_wieght()
        # 节点的左右子节点
        self.left_child = left_child
        self.right_child = right_child


    def isleaf(self):
        """
        基类的方法，返回False，代表是中间节点
        """
        return False

    def get_wieght(self):
        """
        基类的方法，返回对象属性 weight，表示对象的权重
        """
        return self.wieght

    def get_left(self):
        """
        获取左孩子
        """
        return self.left_child

    def get_right(self):
        """
        获取右孩子
        """
        return self.right_child


class HuffTree(object):
    """
    huffTree
    """
    def __init__(self, flag, value =0, freq=0, left_tree=None, right_tree=None):

        super(HuffTree, self).__init__()

        if flag == 0:
            self.root = LeafNode(value, freq)
        else:
            self.root = MiddleNode(left_tree.get_root(), right_tree.get_root())


    def get_root(self):
        """
        获取huffman tree 的根节点
        """
        return self.root

    def get_wieght(self):
        """
        获取这个huffman树的根节点的权重
        """
        return self.root.get_wieght()

    def traverse_huffman_tree(self, root, code, char_freq):
        """
        利用递归的方法遍历huffman_tree，并且以此方式得到每个 字符 对应的huffman编码
        保存在字典 char_freq中
        """
        if root.isleaf():
            char_freq[root.get_value()] = code
            # print(("it = %c  and  freq = %d  code = %s")%(chr(root.get_value()),root.get_wieght(), code))
            return None
        else:
            self.traverse_huffman_tree(root.get_left(), code+'0', char_freq)
            self.traverse_huffman_tree(root.get_right(), code+'1', char_freq)


class HuffmanDecoder:

    def buildHuffmanTree(self,list_hufftrees):
        """
        构造huffman树
        """
        while len(list_hufftrees) >1 :

            # 1. 按照weight 对huffman树进行从小到大的排序
            list_hufftrees.sort(key=lambda x: x.get_wieght())

            # 2. 跳出weight 最小的两个huffman编码树
            temp1 = list_hufftrees[0]
            temp2 = list_hufftrees[1]
            list_hufftrees = list_hufftrees[2:]

            # 3. 构造一个新的huffman树
            newed_hufftree = HuffTree(1, 0, 0, temp1, temp2)

            # 4. 放入到数组当中
            list_hufftrees.append(newed_hufftree)

        # last.  数组中最后剩下来的那棵树，就是构造的Huffman编码树
        return list_hufftrees[0]


    def buildHTree(self, freqAr):
        char_freq = {}
        for i in range(len(freqAr)):
            char_freq[i] = freqAr[i]

        list_hufftrees = []
        for x in char_freq.keys():
            tem = HuffTree(0, x, char_freq[x], None, None)
            list_hufftrees.append(tem)

        tem = self.buildHuffmanTree(list_hufftrees)
        tem.traverse_huffman_tree(tem.get_root(),'',char_freq)

        return tem


    def decompress(self, dataIn, hTree):

        hRoot = hTree.get_root()
        cNode = hRoot
        dataDec = []
        dataInLen = len(dataIn)
        for p in range(dataInLen):
            v = dataIn[p]
            for i in range(8):
                if(v&(1<<(7-i))):
                    cNode = cNode.get_right()
                else:
                    cNode = cNode.get_left()

                if cNode.isleaf():
                    dataDec.append(cNode.get_value())
                    cNode = hRoot

        return dataDec


class VarintDecoder:
    def Decode(self,dataIn):
        dLen = len(dataIn)
        dataDec = []
        offset = 0
        res = 0
        for v in dataIn:
            if (v & 0x80) == 0x80:
                res |= (v & 0x7F) << offset
                offset += 7
            else:
                res |= v << offset
                dataDec.append(res)
                res = 0
                offset = 0

        return dataDec

class ZigZagDecoder:
    def Decode(self,dataIn):
        dLen = len(dataIn)
        dataDec = []
        for v in dataIn:
            res = (v>>1)^(-1*(v&1))
            dataDec.append(res)

        return dataDec

class NDFDecoder:
    def cumsum(self, dataIn, rows, cols):
        for r in range(rows):
            for c in range(cols-1):
                dataIn[r*cols+c+1] += dataIn[r*cols+c]

    def BuildHuffmanTree(self,freqAr):
        hDecoder = HuffmanDecoder()
        hTree = hDecoder.buildHTree(freqAr)
        return hTree

    def HVZD(self,dataIn,huffmanRoot,rows,cols):
        hDecoder = HuffmanDecoder()
        dataHDec = hDecoder.decompress(dataIn,huffmanRoot)
        vDecoder = VarintDecoder()
        dataVDec = vDecoder.Decode(dataHDec)
        zgDecoder = ZigZagDecoder()
        datazDec = zgDecoder.Decode(dataVDec)
        self.cumsum(datazDec,rows,cols)
        dataRes = datazDec
        return dataRes

    def VZD(self,dataIn,rows,cols):
        vDecoder = VarintDecoder()
        dataVDec = vDecoder.Decode(dataIn)
        zgDecoder = ZigZagDecoder()
        datazDec = zgDecoder.Decode(dataVDec)
        self.cumsum(datazDec,rows,cols)
        dataRes = datazDec[0:rows*cols]
        return dataRes









