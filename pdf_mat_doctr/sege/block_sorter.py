

class BlockSorter:
    """
    排序 PDF 页面中的 layoutparser Block，支持单栏/双栏排序。
    """

    def __init__(self, page_width, column_split_ratio=0.5):
        """
        :param page_width: 页面宽度，用于判断左右列分界
        :param column_split_ratio: 分列比例（例如0.5为正中，0.4表示稍偏左分割）
        """
        self.page_width = page_width
        self.split_x = page_width * column_split_ratio

    def sort_blocks(self, blocks):
        """
        对 layoutparser 返回的 blocks 进行排序。

        :param blocks: List[layoutparser.elements.LayoutBlock]
        :return: List[LayoutBlock] 排序后的列表
        """
        left = []
        right = []

        for b in blocks:
            if b.block.x_1 < self.split_x:
                left.append(b)
            else:
                right.append(b)

        # 各列内部按 y_1 排序，若 y 相同再按 x_1
        left_sorted = sorted(left, key=lambda b: (b.block.y_1, b.block.x_1))
        right_sorted = sorted(right, key=lambda b: (b.block.y_1, b.block.x_1))

        return left_sorted + right_sorted
