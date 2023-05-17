initial_data = [
    # 0
    {
        'connected_tiles': [0],
        'neighboring_spots': [1, 13],
        'port': 0
    },
    # 1
    {
        'connected_tiles': [0],
        'neighboring_spots': [0, 2],
        'port': 0
    },
    # 2
    {
        'connected_tiles': [0, 1],
        'neighboring_spots': [1, 3, 11],
        'port': None
    },
    # 3
    {
        'connected_tiles': [1],
        'neighboring_spots': [2, 4],
        'port': 1
    },
    # 4
    {
        'connected_tiles': [1, 2],
        'neighboring_spots': [3, 5, 9],
        'port': 1
    },
    # 5
    {
        'connected_tiles': [2],
        'neighboring_spots': [4, 6],
        'port': None
    },
    # 6
    {
        'connected_tiles': [2],
        'neighboring_spots': [5, 7],
        'port': None
    },
    # 7
    {
        'connected_tiles': [2, 3],
        'neighboring_spots': [6, 8, 24],
        'port': 2
    },
    # 8
    {
        'connected_tiles': [2, 3, 13],
        'neighboring_spots': [7, 9, 21],
        'port': None
    },
    # 9
    {
        'connected_tiles': [1, 2, 13],
        'neighboring_spots': [4, 8, 10],
        'port': None
    },
    # 10
    {
        'connected_tiles': [1, 12, 13],
        'neighboring_spots': [9, 11, 19],
        'port': None
    },
    # 11
    {
        'connected_tiles': [0, 1, 12],
        'neighboring_spots': [2, 10, 12],
        'port': None
    },
    # 12
    {
        'connected_tiles': [0, 11, 12],
        'neighboring_spots': [11, 13, 17],
        'port': None
    },
    # 13
    {
        'connected_tiles': [0, 11],
        'neighboring_spots': [0, 12, 13],
        'port': None
    },
    # 14
    {
        'connected_tiles': [11],
        'neighboring_spots': [13, 15],
        'port': 8
    },
    # 15
    {
        'connected_tiles': [10, 11],
        'neighboring_spots': [14, 16, 37],
        'port': 8
    },
    # 16
    {
        'connected_tiles': [10, 11, 17],
        'neighboring_spots': [15, 17, 34],
        'port': None
    },
    # 17
    {
        'connected_tiles': [11, 12, 17],
        'neighboring_spots': [12, 16, 18],
        'port': None
    },
    # 18
    {
        'connected_tiles': [12, 17, 18],
        'neighboring_spots': [17, 19, 32],
        'port': None
    },
    # 19
    {
        'connected_tiles': [12, 13, 18],
        'neighboring_spots': [10, 18, 20],
        'port': None
    },
    # 20
    {
        'connected_tiles': [13, 14, 18],
        'neighboring_spots': [19, 21, 30],
        'port': None
    },
    # 21
    {
        'connected_tiles': [3, 13, 14],
        'neighboring_spots': [20, 22, 8],
        'port': None
    },
    # 22
    {
        'connected_tiles': [3, 4, 14],
        'neighboring_spots': [21, 23, 28],
        'port': None
    },
    # 23
    {
        'connected_tiles': [3, 4],
        'neighboring_spots': [24, 25],
        'port': None
    },
    # 24
    {
        'connected_tiles': [3],
        'neighboring_spots': [7, 23],
        'port': 2
    },
    # 25
    {
        'connected_tiles': [4],
        'neighboring_spots': [23, 26],
        'port': 3
    },
    # 26
    {
        'connected_tiles': [4],
        'neighboring_spots': [25, 27],
        'port': 3
    },
    # 27
    {
        'connected_tiles': [4, 5],
        'neighboring_spots': [26, 28, 46],
        'port': None
    },
    # 28
    {
        'connected_tiles': [4, 5, 14],
        'neighboring_spots': [22, 27, 29],
        'port': None
    },
    # 29
    {
        'connected_tiles': [5, 14, 15],
        'neighboring_spots': [28, 30, 44],
        'port': None
    },
    # 30
    {
        'connected_tiles': [14, 15, 18],
        'neighboring_spots': [20, 29, 31],
        'port': None
    },
    # 31
    {
        'connected_tiles': [15, 16, 18],
        'neighboring_spots': [30, 32, 42],
        'port': None
    },
    # 32
    {
        'connected_tiles': [16, 17, 18],
        'neighboring_spots': [18, 31, 33],
        'port': None
    },
    # 33
    {
        'connected_tiles': [9, 16, 17],
        'neighboring_spots': [32, 34, 40],
        'port': None
    },
    # 34
    {
        'connected_tiles': [9, 10, 17],
        'neighboring_spots': [33, 35, 16],
        'port': None
    },
    # 35
    {
        'connected_tiles': [9, 10],
        'neighboring_spots': [36, 38],
        'port': 7
    },
    # 36
    {
        'connected_tiles': [10],
        'neighboring_spots': [35, 37],
        'port': None
    },
    # 37
    {
        'connected_tiles': [10],
        'neighboring_spots': [36, 15],
        'port': None
    },
    # 38
    {
        'connected_tiles': [9],
        'neighboring_spots': [35, 39],
        'port': 7
    },
    # 39
    {
        'connected_tiles': [8, 9],
        'neighboring_spots': [38, 40, 53],
        'port': None
    },
    # 40
    {
        'connected_tiles': [8, 9, 16],
        'neighboring_spots': [33, 39, 41],
        'port': None
    },
    # 41
    {
        'connected_tiles': [7, 8, 16],
        'neighboring_spots': [40, 42, 51],
        'port': None
    },
    # 42
    {
        'connected_tiles': [7, 15, 16],
        'neighboring_spots': [41, 43, 31],
        'port': None
    },
    # 43
    {
        'connected_tiles': [6, 7, 15],
        'neighboring_spots': [42, 44, 49],
        'port': None
    },
    # 44
    {
        'connected_tiles': [5, 6, 15],
        'neighboring_spots': [29, 43, 45],
        'port': None
    },
    # 45
    {
        'connected_tiles': [5, 6],
        'neighboring_spots': [46, 47],
        'port': 4
    },
    # 46
    {
        'connected_tiles': [5],
        'neighboring_spots': [27, 45],
        'port': 4
    },
    # 47
    {
        'connected_tiles': [6],
        'neighboring_spots': [45, 48],
        'port': None
    },
    # 48
    {
        'connected_tiles': [6],
        'neighboring_spots': [47, 49],
        'port': None
    },
    # 49
    {
        'connected_tiles': [6, 7],
        'neighboring_spots': [43, 48, 50],
        'port': 5
    },
    # 50
    {
        'connected_tiles': [7],
        'neighboring_spots': [49, 51],
        'port': 5
    },
    # 51
    {
        'connected_tiles': [7, 8],
        'neighboring_spots': [41, 50, 52],
        'port': None
    },
    # 52
    {
        'connected_tiles': [8],
        'neighboring_spots': [51, 53],
        'port': 6
    },
    # 53
    {
        'connected_tiles': [8],
        'neighboring_spots': [39, 52],
        'port': 6
    }
]