:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,3.9763190007502754e-21).
nn('X0',2,1.1644594251521145e-13).
nn('X0',3,7.506514778667554e-20).
nn('X0',4,4.931375115409968e-25).
nn('X0',5,8.894163617406692e-13).
nn('X0',6,2.0691998461885852e-14).
nn('X0',7,1.4316801216773811e-11).
nn('X0',8,7.726980811922565e-14).
nn('X0',9,1.6124220827886532e-15).
nn('X1',0,9.687732926977333e-06).
nn('X1',1,1.097589663601184e-08).
nn('X1',2,0.9999903440475464).
nn('X1',3,4.8308010369702004e-11).
nn('X1',4,2.6282880404112516e-20).
nn('X1',5,9.145872129296198e-15).
nn('X1',6,1.4347904375017206e-12).
nn('X1',7,1.532340226928497e-11).
nn('X1',8,1.8366586463508527e-12).
nn('X1',9,3.0711442668860604e-17).
nn('X2',0,3.14916670518528e-09).
nn('X2',1,9.826109135246952e-07).
nn('X2',2,6.723673322994728e-06).
nn('X2',3,3.2903631108638365e-06).
nn('X2',4,3.548537506503635e-06).
nn('X2',5,5.867054369446123e-06).
nn('X2',6,6.15548998439408e-08).
nn('X2',7,0.000407907908083871).
nn('X2',8,0.9970423579216003).
nn('X2',9,0.0025292469654232264).
nn('X3',0,6.498261409045131e-18).
nn('X3',1,2.1299314308803607e-12).
nn('X3',2,1.706085134278723e-10).
nn('X3',3,5.722137607494515e-12).
nn('X3',4,2.1400681574079705e-11).
nn('X3',5,2.006599331139114e-08).
nn('X3',6,8.932701041526603e-14).
nn('X3',7,6.548290798491507e-07).
nn('X3',8,0.9999982118606567).
nn('X3',9,1.0725066204031464e-06).
nn('X4',0,1.4785460678012896e-07).
nn('X4',1,1.1622092870311462e-06).
nn('X4',2,7.245632787089562e-06).
nn('X4',3,1.8429561805533012e-06).
nn('X4',4,9.886020002625173e-09).
nn('X4',5,2.7361951651982963e-05).
nn('X4',6,6.251921877264977e-05).
nn('X4',7,7.108875433914363e-05).
nn('X4',8,0.9998239874839783).
nn('X4',9,4.77701041745604e-06).
nn('X5',0,2.1875281100425248e-11).
nn('X5',1,7.721862212151098e-12).
nn('X5',2,7.095179305194321e-11).
nn('X5',3,1.1871826472997782e-06).
nn('X5',4,1.071842598321382e-05).
nn('X5',5,9.303088432943696e-08).
nn('X5',6,4.972536054789756e-14).
nn('X5',7,0.0005586257902905345).
nn('X5',8,5.556303861453671e-08).
nn('X5',9,0.9994292855262756).
nn('X6',0,9.439901829243636e-15).
nn('X6',1,2.889459564287835e-15).
nn('X6',2,2.004511304732064e-15).
nn('X6',3,1.8878397160278356e-15).
nn('X6',4,1.570839276644249e-16).
nn('X6',5,5.745029452164374e-11).
nn('X6',6,9.228054984468028e-22).
nn('X6',7,1.0).
nn('X6',8,1.2824220476110007e-19).
nn('X6',9,1.612729150224368e-08).

a :- Pos=[f(['X0','X1','X2'],10),f(['X3','X4','X5','X6'],32)], metaabd(Pos).