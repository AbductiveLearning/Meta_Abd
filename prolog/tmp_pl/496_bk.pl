:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.408790666346249e-07).
nn('X0',1,0.9999998807907104).
nn('X0',2,2.4463142622721534e-08).
nn('X0',3,3.5391964755653106e-17).
nn('X0',4,2.0794750643648996e-10).
nn('X0',5,1.456150061640571e-10).
nn('X0',6,7.373331112336956e-11).
nn('X0',7,4.4407848065475264e-08).
nn('X0',8,2.941457494243238e-11).
nn('X0',9,1.2500854518204818e-10).
nn('X1',0,0.0007526198169216514).
nn('X1',1,0.07301802188158035).
nn('X1',2,0.6217705607414246).
nn('X1',3,0.26571178436279297).
nn('X1',4,8.90588307811413e-06).
nn('X1',5,0.00234812730923295).
nn('X1',6,0.00038458945346064866).
nn('X1',7,0.007270836271345615).
nn('X1',8,0.028554579243063927).
nn('X1',9,0.000179946786374785).
nn('X2',0,1.1909468412341084e-05).
nn('X2',1,0.9999871253967285).
nn('X2',2,1.514540031166689e-07).
nn('X2',3,6.223865281658469e-13).
nn('X2',4,1.2960541084794386e-07).
nn('X2',5,2.6418203447065025e-07).
nn('X2',6,3.2494739343746915e-08).
nn('X2',7,3.2776395642031275e-07).
nn('X2',8,2.6358382143598647e-09).
nn('X2',9,2.1278040662764397e-08).
nn('X3',0,0.8684903383255005).
nn('X3',1,5.681654036138184e-10).
nn('X3',2,5.6626275181770325e-05).
nn('X3',3,7.249115197005551e-10).
nn('X3',4,1.7413606201444054e-07).
nn('X3',5,0.00321406708098948).
nn('X3',6,0.12804186344146729).
nn('X3',7,5.5728293091306114e-08).
nn('X3',8,0.00019671542395371944).
nn('X3',9,8.534259166026459e-08).
nn('X4',0,1.560026066727005e-05).
nn('X4',1,0.9999774694442749).
nn('X4',2,5.03437490806391e-07).
nn('X4',3,9.739385513352072e-12).
nn('X4',4,5.686620170308743e-06).
nn('X4',5,3.5383033036850975e-07).
nn('X4',6,2.8498661208686826e-07).
nn('X4',7,6.473158009612234e-08).
nn('X4',8,2.265972076642697e-09).
nn('X4',9,3.213234478494087e-08).
nn('X5',0,8.697271880464541e-08).
nn('X5',1,1.7750302504282445e-05).
nn('X5',2,6.192712316988036e-05).
nn('X5',3,0.0034371409565210342).
nn('X5',4,0.0004328259383328259).
nn('X5',5,0.01497392263263464).
nn('X5',6,2.2155045371619053e-05).
nn('X5',7,0.002663057530298829).
nn('X5',8,0.9761239290237427).
nn('X5',9,0.002267052186653018).

a :- Pos=[f(['X0','X1','X2','X3'],5),f(['X4','X5'],9)], metaabd(Pos).