:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.2048162584943978e-14).
nn('X0',1,1.2225843167706918e-13).
nn('X0',2,8.165129949120123e-15).
nn('X0',3,5.0822756105569494e-15).
nn('X0',4,2.6376519123534594e-18).
nn('X0',5,5.4437184604906944e-14).
nn('X0',6,3.536361379226183e-25).
nn('X0',7,1.0).
nn('X0',8,1.912837071952977e-20).
nn('X0',9,9.023631841742485e-10).
nn('X1',0,3.8394588841583754e-07).
nn('X1',1,0.9999996423721313).
nn('X1',2,9.466466499574722e-10).
nn('X1',3,1.2755635416005632e-17).
nn('X1',4,1.2267492577322514e-09).
nn('X1',5,5.418897552900148e-10).
nn('X1',6,3.8325991269516635e-09).
nn('X1',7,2.8154444053907213e-11).
nn('X1',8,1.6547663846816496e-12).
nn('X1',9,9.712190253419184e-12).
nn('X2',0,1.1508471953192156e-09).
nn('X2',1,3.9684606136787855e-18).
nn('X2',2,8.539431098006367e-13).
nn('X2',3,2.936075774799382e-19).
nn('X2',4,1.3953413077749666e-12).
nn('X2',5,3.2579657727183076e-07).
nn('X2',6,0.9999996423721313).
nn('X2',7,1.786000944779451e-15).
nn('X2',8,2.494352651944559e-13).
nn('X2',9,8.38288407408696e-19).
nn('X3',0,4.009432230667187e-10).
nn('X3',1,2.918419506059422e-09).
nn('X3',2,1.1887868467397311e-08).
nn('X3',3,1.0418572254922154e-10).
nn('X3',4,5.4683770532582e-12).
nn('X3',5,5.4063000604065437e-11).
nn('X3',6,7.395477749701798e-17).
nn('X3',7,0.999998927116394).
nn('X3',8,1.3767165304903045e-13).
nn('X3',9,1.1195536444574827e-06).
nn('X4',0,1.7074067160116613e-10).
nn('X4',1,5.695530174874494e-14).
nn('X4',2,4.503082802476133e-12).
nn('X4',3,3.50167256302214e-11).
nn('X4',4,3.22597612685549e-12).
nn('X4',5,1.0).
nn('X4',6,3.905304968299106e-10).
nn('X4',7,1.1307806774274454e-10).
nn('X4',8,2.1776245737181732e-11).
nn('X4',9,1.2659849835117143e-09).
nn('X5',0,2.050898784133892e-09).
nn('X5',1,2.7488579235068755e-06).
nn('X5',2,5.649234299198724e-05).
nn('X5',3,0.9999187588691711).
nn('X5',4,1.8693636319889323e-13).
nn('X5',5,2.1905076209804974e-05).
nn('X5',6,1.4416302556558534e-14).
nn('X5',7,1.343613575954805e-07).
nn('X5',8,9.90220197505387e-11).
nn('X5',9,9.130029371418047e-11).

a :- Pos=[f(['X0','X1','X2'],14),f(['X3','X4','X5'],15)], metaabd(Pos).
