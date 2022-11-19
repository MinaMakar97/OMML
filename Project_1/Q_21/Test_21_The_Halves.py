from function_21_The_Halves import * 
def ICanGeneralize(x):
    W = np.array([[-0.40431947 ,-1.37819615 , 0.99378432]
    , [ 1.93096316, -1.19236937,  1.86366418]
    , [-1.22262065,  1.27638763, -0.43574854]
    , [-1.55733773,  0.49660274,  0.893181  ]
    , [-1.94620305, -1.81888412,  0.9142164 ]
    , [-1.56974757, -0.02010056, -0.3123172 ]
    , [-0.81571662,  0.03343687, -0.30233842]
    , [-0.90037241,  1.48874275, -1.90775362]
    , [ 1.28263035,  1.41114273,  1.54386313]
    , [ 1.00790153, -0.16031381, -0.75234853]
    , [-0.86994632, -1.29027242, -0.49666107]
    , [-0.70760183,  0.69087178, -0.50786434]
    , [-0.62349389,  1.45805875, -1.48437458]
    , [ 0.11531616, -0.90548944, -0.29567367]
    , [-1.52865136,  1.74224632,  1.17484738]
    , [-0.91984197, -1.79396689, -1.0077943 ]
    , [-1.42629718,  0.70866957, -0.85808791]
    , [-0.71389335,  0.09237176, -1.88689237]
    , [-0.37786688, -1.86352867,  1.93302   ]
    , [-0.99901606,  1.64572026,  1.09072579]
    , [-0.28723505,  1.81152488, -1.94863964]
    , [ 0.29538393, -0.96249456, -1.63248274]
    , [ 1.89971266, -1.4418105 ,  1.96281345]
    , [-0.88849641,  0.26459603,  0.65980685]
    , [ 1.02980788,  0.16122102, -1.44276788]
    , [-1.90789236, -0.08327629,  0.51644313]
    , [ 1.54349908,  1.92742923,  1.83315348]
    , [-0.71842482,  0.42560422, -0.75813088]
    , [-1.00636401, -0.96822874, -1.11033839]
    , [ 0.82881605, -0.64585589,  0.3649881 ]
    , [-1.04814616, -0.94687189,  1.74911302]
    , [ 1.93543512, -1.79192189, -0.52878387]
    , [-0.79546369,  1.38807867, -1.61424459]
    , [ 1.66146673, -0.72521158,  0.62952169]
    , [-1.78751283,  0.16363637,  0.68995944]
    , [ 0.07193544,  1.10577339, -0.06854224]
    , [ 1.16710969,  0.33395262, -0.85684407]
    , [ 0.37610381, -1.2983438 , -1.74222999]
    , [ 0.9308578 ,  0.69839196,  0.51022562]
    , [-0.74301685,  0.62105065, -1.1049486 ]
    , [-1.19799353,  0.27694806,  0.97632564]
    , [-0.82485513, -1.60805934,  1.91320092]
    , [ 0.90701128, -0.09509857,  1.85896025]
    , [-0.64020234, -1.33602685, -0.10410654]
    , [-1.47616149, -1.26060696, -0.15385766]
    , [ 0.27920941, -0.03986631,  1.18329042]
    , [-1.95310988, -1.75228595,  1.85342602]
    , [ 0.54035418,  1.91959333, -1.04201617]
    , [-1.70422754, -1.90483318,  0.03312345]
    , [-0.55008497,  0.543295  ,  1.20851784]])
 
    V = np.array(
    [[ -0.03647255 , -0.6284004  ,  0.05676286 , -4.16205474,  -0.39240448
    , 4.68517993 ,-13.30078831 ,  2.0605789  , -1.82058318,  -7.95273318
    ,-0.16001834 ,  1.93560996 ,  0.92258877 , -3.27600828,  -0.26578613
    , 1.31284996 ,  1.9798804  , -4.65348301 , -1.06648045,  -0.20989431
    ,-0.08173921 , -0.52498726 , -0.73402352 , -7.53935064,  -0.24661031
    , 3.04996109 ,  0.38009508 ,  2.07413974 , -2.54850017,   2.97301834
    , 0.3226927  , -0.59900946 , -1.14443281 , -2.67433096,  -2.41332204
    ,-0.07986098 , -1.03789708 ,  0.88262758 , -2.68518637,  -6.09276253
    , 2.65040545 ,  0.29343242 , -4.70336065 ,  0.85051145,  -1.6264988
    ,-3.14078586 , -0.80314286 , -0.16932091 ,  1.46017832,   0.8394225 ]])

    x_trans = transform(x)
    G = transform_G(W, x_trans)
    return feedforward(G, V)