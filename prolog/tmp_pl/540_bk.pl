:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.00016076743486337364).
nn('X0',1,0.9994256496429443).
nn('X0',2,0.00019660325779113919).
nn('X0',3,5.4563756179959455e-08).
nn('X0',4,1.5999645256670192e-05).
nn('X0',5,9.434621460968629e-05).
nn('X0',6,9.142648195847869e-05).
nn('X0',7,1.4094882317294832e-05).
nn('X0',8,7.053384933897178e-07).
nn('X0',9,3.349257156060048e-07).
nn('X1',0,2.6398316865794413e-10).
nn('X1',1,1.1367578736098949e-06).
nn('X1',2,0.0001088403514586389).
nn('X1',3,0.9998875856399536).
nn('X1',4,1.9874225420761504e-14).
nn('X1',5,2.4219216356868856e-06).
nn('X1',6,4.897708613657895e-16).
nn('X1',7,1.1625139961779496e-08).
nn('X1',8,4.10980659226734e-11).
nn('X1',9,1.3810204715913876e-11).
nn('X2',0,5.417994231038392e-08).
nn('X2',1,4.566877009892778e-07).
nn('X2',2,1.9712131926752363e-09).
nn('X2',3,2.555420763883376e-08).
nn('X2',4,2.187484415827612e-09).
nn('X2',5,0.9999960064888).
nn('X2',6,3.48569869856874e-06).
nn('X2',7,5.474532382976349e-09).
nn('X2',8,2.159328937878513e-09).
nn('X2',9,1.538106586540522e-10).
nn('X3',0,8.580198725995081e-13).
nn('X3',1,3.8972751620954676e-14).
nn('X3',2,1.5964785049504826e-11).
nn('X3',3,9.697105696204744e-08).
nn('X3',4,4.1806390072451904e-05).
nn('X3',5,2.0031778547036083e-07).
nn('X3',6,8.097169954663069e-15).
nn('X3',7,0.0018547512590885162).
nn('X3',8,1.3702897150302817e-10).
nn('X3',9,0.9981032013893127).
nn('X4',0,8.585681463324748e-17).
nn('X4',1,3.386494050813685e-15).
nn('X4',2,5.283112575125542e-17).
nn('X4',3,5.1809397842099055e-20).
nn('X4',4,2.7435989565057044e-21).
nn('X4',5,2.043896600361847e-13).
nn('X4',6,3.795914551303538e-26).
nn('X4',7,1.0).
nn('X4',8,2.7945059836379295e-25).
nn('X4',9,3.590930938018877e-12).

a :- Pos=[f(['X0','X1','X2','X3','X4'],25)], metaabd(Pos).
