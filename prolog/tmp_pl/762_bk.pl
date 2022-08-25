:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.9833442568778992).
nn('X0',1,0.008285853080451488).
nn('X0',2,0.0001737763377605006).
nn('X0',3,2.3215263809106546e-06).
nn('X0',4,1.2290704034967348e-05).
nn('X0',5,0.0003555583825800568).
nn('X0',6,0.0003210735449101776).
nn('X0',7,0.007007358595728874).
nn('X0',8,0.0003011349181178957).
nn('X0',9,0.0001964216644410044).
nn('X1',0,2.461645863149897e-06).
nn('X1',1,2.923312877101125e-06).
nn('X1',2,0.9999940991401672).
nn('X1',3,3.247671287454068e-08).
nn('X1',4,5.43582241935639e-14).
nn('X1',5,2.0920373072774723e-11).
nn('X1',6,9.39445483211232e-12).
nn('X1',7,5.210784479459107e-07).
nn('X1',8,1.6111211920133428e-08).
nn('X1',9,2.253475010760564e-11).
nn('X2',0,1.900379054120549e-08).
nn('X2',1,1.0).
nn('X2',2,2.72524003364083e-10).
nn('X2',3,1.1263025637371565e-19).
nn('X2',4,5.135916106464855e-13).
nn('X2',5,1.0216102963589435e-10).
nn('X2',6,7.221800935042211e-11).
nn('X2',7,6.857632794954327e-10).
nn('X2',8,2.191823653997782e-13).
nn('X2',9,4.779773040038127e-13).
nn('X3',0,2.4093477435371824e-08).
nn('X3',1,4.228543293294251e-09).
nn('X3',2,2.756059132025257e-07).
nn('X3',3,0.00013324973406270146).
nn('X3',4,0.00041696414700709283).
nn('X3',5,5.764392335549928e-05).
nn('X3',6,5.5621542927042356e-09).
nn('X3',7,0.011936800554394722).
nn('X3',8,0.00011713994899764657).
nn('X3',9,0.9873380064964294).
nn('X4',0,6.335034754556546e-07).
nn('X4',1,8.859459173227169e-12).
nn('X4',2,0.00011355531023582444).
nn('X4',3,2.2661283889803096e-16).
nn('X4',4,0.910801351070404).
nn('X4',5,7.350893429247662e-05).
nn('X4',6,0.08900908380746841).
nn('X4',7,4.768179739933487e-10).
nn('X4',8,1.93020321948012e-11).
nn('X4',9,1.879414526229084e-06).
nn('X5',0,1.9827471078315284e-06).
nn('X5',1,0.9999736547470093).
nn('X5',2,2.2508744223159738e-05).
nn('X5',3,2.2830624543668243e-11).
nn('X5',4,1.670618530624779e-06).
nn('X5',5,1.3028010670268486e-08).
nn('X5',6,1.237858526792479e-07).
nn('X5',7,1.3825327549454869e-08).
nn('X5',8,4.6567080858039844e-08).
nn('X5',9,2.6918440809708954e-08).
nn('X6',0,3.896917769452557e-05).
nn('X6',1,1.6471698245368316e-06).
nn('X6',2,0.9999592304229736).
nn('X6',3,3.7495659355180067e-11).
nn('X6',4,5.430935970986974e-16).
nn('X6',5,2.3402404146499745e-12).
nn('X6',6,3.6112929313603104e-10).
nn('X6',7,8.592536460128031e-08).
nn('X6',8,3.0127822459036224e-10).
nn('X6',9,2.9521616271357964e-13).

a :- Pos=[f(['X0','X1','X2'],3),f(['X3','X4','X5','X6'],16)], metaabd(Pos).
