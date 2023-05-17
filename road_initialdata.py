initial_data_roads = [
    # 0
    {
        'neighboring_roads': [1,5],
        'neighboring_spots': [0, 1],
        'road': None
    },
    # 1
    {
        'neighboring_roads': [0,6],
        'neighboring_spots': [1, 2],
        'road': None
    },
    # 2
    {
        'neighboring_roads': [1,3,6,10],
        'neighboring_spots': [2, 11],
        'road': None
    },
    # 3
    {
        'neighboring_roads': [2,4,19,26],
        'neighboring_spots': [11, 12],
        'road': None
    },
    # 4
    {
        'neighboring_roads': [3,5,26,30],
        'neighboring_spots': [12, 13],
        'road': None
    },
    # 5
    {
        'neighboring_roads': [0,4,30],
        'neighboring_spots': [13, 0],
        'road': None
    },
    # 6
    {
        'neighboring_roads': [1,2,7],
        'neighboring_spots': [2, 3],
        'road': None
    },
    # 7
    {
        'neighboring_roads': [6,8,11],
        'neighboring_spots': [3, 4],
        'road': None
    },
    # 8
    {
        'neighboring_roads': [7,0,11,15],
        'neighboring_spots': [4, 9],
        'road': None
    },
    # 9
    {
        'neighboring_roads': [8,19,15,23],
        'neighboring_spots': [9, 10],
        'road': None
    },
    # 10
    {
        'neighboring_roads': [2,3,9,23],
        'neighboring_spots': [10, 11],
        'road': None
    },
    # 11
    {
        'neighboring_roads': [7,8,12],
        'neighboring_spots': [4, 5],
        'road': None
    },
    # 12
    {
        'neighboring_roads': [11,13],
        'neighboring_spots': [5, 6],
        'road': None
    },
    # 13
    {
        'neighboring_roads': [12,14,16],
        'neighboring_spots': [6, 7],
        'road': None
    },
    # 14
    {
        'neighboring_roads': [13,15,16,20],
        'neighboring_spots': [7, 8],
        'road': None
    },
    # 15
    {
        'neighboring_roads': [8,9,14,20],
        'neighboring_spots': [8, 9],
        'road': None
    },
    # 16
    {
        'neighboring_roads': [13,14,17],
        'neighboring_spots': [7, 24],
        'road': None
    },
    # 17
    {
        'neighboring_roads': [16,18,48],
        'neighboring_spots': [24, 23],
        'road': None
    },
    # 18
    {
        'neighboring_roads': [17,19,44,48],
        'neighboring_spots': [23, 22],
        'road': None
    },
    # 19
    {
        'neighboring_roads': [18,20,21,44],
        'neighboring_spots': [22, 21],
        'road': None
    },
    # 20
    {
        'neighboring_roads': [14,15,19,21],
        'neighboring_spots': [21, 8],
        'road': None
    },
    # 21
    {
        'neighboring_roads': [19,20,22,41],
        'neighboring_spots': [21, 20],
        'road': None
    },
    # 22
    {
        'neighboring_roads': [21,23,,24,41],
        'neighboring_spots': [20, 19],
        'road': None
    },
    # 23
    {
        'neighboring_roads': [9,10,22,24],
        'neighboring_spots': [19, 10],
        'road': None
    },
    # 24
    {
        'neighboring_roads': [22,23,25,38],
        'neighboring_spots': [19, 18],
        'road': None
    },
    # 25
    {
        'neighboring_roads': [24,26,27,38],
        'neighboring_spots': [18, 17],
        'road': None
    },
    # 26
    {
        'neighboring_roads': [3,4,25,27],
        'neighboring_spots': [17, 12],
        'road': None
    },
    # 27
    {
        'neighboring_roads': [25,26,28,35],
        'neighboring_spots': [17, 16],
        'road': None
    },
    # 28
    {
        'neighboring_roads': [27,29,31,35],
        'neighboring_spots': [15, 16],
        'road': None
    },
    # 29
    {
        'neighboring_roads': [28,30,31],
        'neighboring_spots': [13,14 ],
        'road': None
    },
    # 30
    {
        'neighboring_roads': [4,5,29],
        'neighboring_spots': [14,13 ],
        'road': None
    },
    # 31
    {
        'neighboring_roads': [28,29,31],
        'neighboring_spots': [15, 37],
        'road': None
    },
    # 32
    {
        'neighboring_roads': [31,33],
        'neighboring_spots': [36, 37],
        'road': None
    },
    # 33
    {
        'neighboring_roads': [31,34,61],
        'neighboring_spots': [35,36 ],
        'road': None
    },
    # 34
    {
        'neighboring_roads': [33,35,36,61],
        'neighboring_spots': [34, 35],
        'road': None
    },
    # 35
    {
        'neighboring_roads': [27,28,34,36],
        'neighboring_spots': [16,34 ],
        'road': None
    },
    # 36
    {
        'neighboring_roads': [34,35,37,58],
        'neighboring_spots': [33,34 ],
        'road': None
    },
    # 37
    {
        'neighboring_roads': [36,38,39,58],
        'neighboring_spots': [32,33 ],
        'road': None
    },
    # 38
    {
        'neighboring_roads': [24,25,37,39],
        'neighboring_spots': [1, 13],
        'road': None
    },
    # 39
    {
        'neighboring_roads': [37,38,40,55],
        'neighboring_spots': [31, 32],
        'road': None
    },
    # 40
    {
        'neighboring_roads': [39,41,42,55],
        'neighboring_spots': [30, 31 ],
        'road': None
    },
    # 41
    {
        'neighboring_roads': [22,21,40,42],
        'neighboring_spots': [20,30 ],
        'road': None
    },
    # 42
    {
        'neighboring_roads': [40,41,43,54],
        'neighboring_spots': [29, 30],
        'road': None
    },
    # 43
    {
        'neighboring_roads': [42,44,45,52],
        'neighboring_spots': [28, 29],
        'road': None
    },
    # 44
    {
        'neighboring_roads': [18,19,43,45],
        'neighboring_spots': [22, 28],
        'road': None
    },
    # 45
    {
        'neighboring_roads': [43,44,46,49],
        'neighboring_spots': [28,27 ],
        'road': None
    },
    # 46
    {
        'neighboring_roads': [45,47,49],
        'neighboring_spots': [26, 27],
        'road': None
    },
    # 47
    {
        'neighboring_roads': [46,48],
        'neighboring_spots': [25, 26],
        'road': None
    },
    # 48
    {
        'neighboring_roads': [17,18,47],
        'neighboring_spots': [25, 28],
        'road': None
    },
    # 49
    {
        'neighboring_roads': [45,46,50],
        'neighboring_spots': [27, 46],
        'road': None
    },
    # 50
    {
        'neighboring_roads': [49,51,71],
        'neighboring_spots': [45, 46],
        'road': None
    },
    # 51
    {
        'neighboring_roads': [50,52,53,71],
        'neighboring_spots': [44, 45],
        'road': None
    },
    # 52
    {
        'neighboring_roads': [42,43,51,53],
        'neighboring_spots': [29,44 ],
        'road': None
    },
    # 53
    {
        'neighboring_roads': [51,52,54,68],
        'neighboring_spots': [43,44 ],
        'road': None
    }
     # 54
    {
        'neighboring_roads': [53,55,56,68],
        'neighboring_spots': [42, 43],
        'road': None
    }
     # 55
    {
        'neighboring_roads': [39,40,54,56],
        'neighboring_spots': [31, 42],
        'road': None
    }
     # 56
    {
        'neighboring_roads': [54,55,57,65],
        'neighboring_spots': [41, 42],
        'road': None
    }
     # 57
    {
        'neighboring_roads': [56,58,59,65],
        'neighboring_spots': [40, 41],
        'road': None
    }
     # 58
    {
        'neighboring_roads': [36,37,57,59],
        'neighboring_spots': [33, 40],
        'road': None
    }
     # 59
    {
        'neighboring_roads': [57,58,60,62],
        'neighboring_spots': [39, 40],
        'road': None
    }
     # 60
    {
        'neighboring_roads': [59,61,62],
        'neighboring_spots': [38, 39],
        'road': None
    }
    
     # 61
    {
        'neighboring_roads': [33,34,60],
        'neighboring_spots': [35, 38],
        'road': None
    }
     # 62
    {
        'neighboring_roads': [59,60,63],
        'neighboring_spots': [39, 53],
        'road': None
    }
     # 63
    {
        'neighboring_roads': [62,64],
        'neighboring_spots': [52, 53],
        'road': None
    }
     # 64
    {
        'neighboring_roads': [63,65,66],
        'neighboring_spots': [51, 52],
        'road': None
    }
     # 65
    {
        'neighboring_roads': [56,57,64,66],
        'neighboring_spots': [41, 51],
        'road': None
    }
     # 66
    {
        'neighboring_roads': [64,65,67,58],
        'neighboring_spots': [51, 50],
        'road': None
    }
     # 67
    {
        'neighboring_roads': [66,68,69],
        'neighboring_spots': [49, 50],
        'road': None
    }
     # 68
        'neighboring_roads': [53,54,67,69],
        'neighboring_spots': [43, 49],
        'road': None
    }
     # 69
    {
        'neighboring_roads': [67,68,70],
        'neighboring_spots': [48, 49],
        'road': None
    }
     # 70
    {
        'neighboring_roads': [69,71],
        'neighboring_spots': [47, 48],
        'road': None
    }
    # 71
    {
        'neighboring_roads': [50,51,70],
        'neighboring_spots': [45, 47],
        'road': None
    }
]
