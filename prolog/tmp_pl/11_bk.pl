:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.9999993443489075).
nn('X0',1,1.0432086365745933e-15).
nn('X0',2,6.147427029645769e-07).
nn('X0',3,2.4858639926790375e-14).
nn('X0',4,4.3914166393537065e-16).
nn('X0',5,7.002470270300876e-10).
nn('X0',6,2.050641612072468e-08).
nn('X0',7,1.3390186529016468e-11).
nn('X0',8,1.2713169059153095e-11).
nn('X0',9,1.1422010256015303e-12).
nn('X1',0,1.0068722076539416e-05).
nn('X1',1,1.3873642501494032e-06).
nn('X1',2,8.757485920796171e-05).
nn('X1',3,1.7290843743467121e-06).
nn('X1',4,5.567015364249528e-07).
nn('X1',5,4.581593202601653e-06).
nn('X1',6,0.00017182451847475022).
nn('X1',7,1.819039971451275e-05).
nn('X1',8,0.999647855758667).
nn('X1',9,5.6220818805741146e-05).
nn('X2',0,7.612590593453206e-07).
nn('X2',1,7.183320605008703e-11).
nn('X2',2,4.611099946316699e-09).
nn('X2',3,2.0820318802350357e-09).
nn('X2',4,4.3376743974476994e-08).
nn('X2',5,0.999991774559021).
nn('X2',6,7.332943823712412e-06).
nn('X2',7,1.4685264204672421e-07).
nn('X2',8,1.2667417337297593e-08).
nn('X2',9,3.046421426233792e-08).
nn('X3',0,1.3255696995791155e-13).
nn('X3',1,1.2292048956930209e-15).
nn('X3',2,8.096344233064767e-13).
nn('X3',3,2.9155000191849467e-08).
nn('X3',4,4.101600268313632e-07).
nn('X3',5,2.4468682635614414e-09).
nn('X3',6,6.318452196210804e-17).
nn('X3',7,0.000308036629576236).
nn('X3',8,1.2486056633065346e-10).
nn('X3',9,0.9996916055679321).
nn('X4',0,1.580021944391774e-07).
nn('X4',1,1.1202221372741406e-08).
nn('X4',2,9.363566277897917e-07).
nn('X4',3,7.736433326499537e-05).
nn('X4',4,0.0001090829900931567).
nn('X4',5,1.789021325748763e-06).
nn('X4',6,5.202860919872876e-10).
nn('X4',7,0.01155195850878954).
nn('X4',8,8.068697206908837e-05).
nn('X4',9,0.9881780743598938).
nn('X5',0,7.92057131687729e-12).
nn('X5',1,6.7646789148057e-07).
nn('X5',2,9.359093837701948e-07).
nn('X5',3,0.999886155128479).
nn('X5',4,1.5132756904664112e-14).
nn('X5',5,0.00011225204798392951).
nn('X5',6,1.4646021544683486e-16).
nn('X5',7,1.0929915639223964e-08).
nn('X5',8,3.98014466437141e-13).
nn('X5',9,1.0281549917001698e-10).

a :- Pos=[f(['X0','X1','X2','X3'],22),f(['X4','X5'],12)], metaabd(Pos).