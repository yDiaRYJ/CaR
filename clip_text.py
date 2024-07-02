
BACKGROUND_CATEGORY = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign',
                        ]

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
                   ]
                   
new_class_names = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair seat', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
                   ]


class_names_coco = ['person','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack',
                    'umbrella','handbag','tie','suitcase','frisbee',
                    'skis','snowboard','sports ball','kite','baseball bat',
                    'baseball glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','spoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair','sofa','pottedplant','bed',
                    'diningtable','toilet','tvmonitor','laptop','mouse',
                    'remote','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hair drier','toothbrush',
]

new_class_names_coco = ['person with clothes,people,human','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird avian',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack,bag',
                    'umbrella,parasol','handbag,purse','necktie','suitcase','frisbee',
                    'skis','sknowboard','sports ball','kite','baseball bat',
                    'glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','dessertspoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair seat','sofa','pottedplant','bed',
                    'diningtable','toilet','tvmonitor screen','laptop','mouse',
                    'remote control','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hairdrier,blowdrier','toothbrush',
                    ]


BACKGROUND_CATEGORY_COCO = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge',
                        ]

# 陆地
BACKGROUND_Terrestrial = ['ground','land','grass','tree','mountain','rock','valley','earth','terrain','forest','bush','hill','field','pasture',
                          'meadow','plateau','cliff','canyon','ridge','peak','plain','prairie','tundra','savanna','steppe','crag','knoll','dune',
                          'glen','dale','copse','thicket']
# 水生、大气
BACKGROUND_Aquatic_Atmospheric = ['sea','ocean','lake','water','river','sky','cloud','pond','stream','lagoon','bay','gulf','fjord','estuary',
                                  'creek','brook','reservoir','pool','spring','marsh','swamp','wetland','glacier','iceberg','atmosphere',
                                  'stratosphere','mist','fog','rain','drizzle','hail','sleet','snow','thunderstorm','breeze','wind','gust',
                                  'hurricane','tornado','monsoon','cumulus','cirrus','stratus','nimbus']
# 人造物
BACKGROUND_Man_Made = ['building','house','wall','road','street','railway','railroad','bridge','edifice','structure','apartment','condominium',
                       'skyscraper','highway','boulevard','lane','alley','byway','avenue','expressway','freeway','path','overpass','underpass',
                       'viaduct','tunnel','footbridge','crosswalk','culvert','dam','archway','causeway','plaza','square','station','terminal']