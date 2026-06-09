'''
Jet engine by Long Yizhu (2023.01)
'''

import numpy as np
import pandas as pd
import math

# NILS: added to prevent `FileNotFoundError: [Errno 2] No such file or directory: './dump/engine_outward.dat'`
import sys
sys.path.append('.')
import matplotlib.pyplot as plt

from cst_modeling.basic import BasicSection, BasicSurface
from cst_modeling.section import cst_curve, bump_function
from cst_modeling.section import RoundTipSection

# =============================================================================
import sys
sys.path.append(r'C:\Users\nmb48\Documents\GitHub\cst-modeling3d\nils')
# # from upstream_duct_final import get_duct_geometry
# # from upstream_duct_final_test import get_duct_geometry
# # from upstream_duct_final_matched_slopes import get_duct_geometry
# from upstream_duct_final_combined import get_duct_geometry

from radial_ducted_radiator.annular_diffuser_geometry import parametrize_diffuser_v1
# sys.exit()


sys.path.append(r'C:\Users\nmb48\Documents\GitHub\Flydrogen')
from flydrogen.components import AnnularDiffuserModel

ANNULARDIFFUSERMODEL = AnnularDiffuserModel()

# sys.exit()
# =============================================================================


class Section(BasicSection):

    def __init__(self, n_curve = 4):
        self.x = np.zeros(1)
        self.y = np.zeros(1)
        self.z = np.zeros(1)
        self.n_ = max(1,n_curve)
        self.curve = [curve() for _ in range(self.n_)]

    def set_Section(self, sec_x, sec_y, sec_z):
        self.x = sec_x
        self.y = sec_y
        self.z = sec_z

    def join_curve(self):
        for i in range(self.n_):
            self.x = np.append(self.x, self.curve[i].curve_x)
            self.y = np.append(self.y, self.curve[i].curve_y)
            self.z = np.append(self.z, self.curve[i].curve_z)

        self.x = np.delete(self.x,0)
        self.y = np.delete(self.y,0)
        self.z = np.delete(self.z,0)

        self.x,self.y,self.z = order_curve3d(self.x,self.y,self.z)

        self.x[0] = self.x[-1]
        self.y[0] = self.y[-1]
        self.z[0] = self.z[-1]

    def circle(self, p1=np.array([1,0]), p2=np.array([0,1]), p3=np.array([0,-1]), nn=500):
        A = p1[0]*(p2[1]-p3[1]) - p1[1]*(p2[0]-p3[0]) + p2[0]*p3[1] - p3[0]*p2[1]
        if np.abs(A) <= 1E-20:
            raise Exception('Finding circle: 3 points in one line')
        
        p1s = p1[0]**2 + p1[1]**2
        p2s = p2[0]**2 + p2[1]**2
        p3s = p3[0]**2 + p3[1]**2

        B = p1s*(p3[1]-p2[1]) + p2s*(p1[1]-p3[1]) + p3s*(p2[1]-p1[1])
        C = p1s*(p2[0]-p3[0]) + p2s*(p3[0]-p1[0]) + p3s*(p1[0]-p2[0])
        D = p1s*(p3[0]*p2[1]-p2[0]*p3[1]) + p2s*(p1[0]*p3[1]-p3[0]*p1[1]) + p3s*(p2[0]*p1[1]-p1[0]*p2[1])

        cen_x = -B/2/A
        cen_y = -C/2/A

        R = np.sqrt(B**2+C**2-4*A*D)/2/np.abs(A)

        for i in range(nn):
            theta = i*2*np.pi/(nn-1)
            self.x = np.append(self.x, cen_x + R*math.cos(theta))
            self.y = np.append(self.y, cen_y + R*math.sin(theta))
        self.x = np.delete(self.x,0)
        self.y = np.delete(self.y,0)


    def order_curve3d(x,y,z,axis = 'z'):
        if axis == 'x':
            x1 = y
            x2 = z
            x3 = x
        if axis == 'y':
            x1 = x
            x2 = z
            x3 = y
        if axis == 'z':  
            x1 = x
            x2 = y
            x3 = z

        cen_x = np.mean(x1)
        cen_y = np.mean(x2)
        
        x1s = []
        x2s = []
        x3s = []
        for i in range(len(x1)):
            dx = x1[i] - cen_x
            dy = x2[i] - cen_y
            ang = np.arctan2(dy,dx)
            if ang < 0:
                ang += np.pi*2
            x1s.append([x1[i],ang])
            x2s.append([x2[i],ang])
            x3s.append([x3[i],ang])

        x1s = sorted(x1s, key = lambda x:x[1])
        x2s = sorted(x2s, key = lambda x:x[1])
        x3s = sorted(x3s, key = lambda x:x[1])

        x1 = np.array([x[0] for x in x1s])
        x2 = np.array([x[0] for x in x2s])
        x3 = np.array([x[0] for x in x3s])

        if axis == 'x':
            return x3,x1,x2
        if axis == 'y':
            return x1,x3,x2
        if axis == 'z':  
            return x1,x2,x3

class curve():

    def __init__(self, r0=np.array([0,0,0]), r1=np.array([1,0,0]), normal = np.array([0,0,1]), nn=125, coef = np.array([0,0,0,0]), xn1=0.75, xn2=1.0):
        #确定曲线增加的方向
        dr = r1 - r0
        D = np.cross(normal,dr)
        D = D / np.linalg.norm(D)

        #确定基本直线
        L0 = np.linalg.norm(dr)
        base_x = np.zeros(nn)
        base_y = np.zeros(nn)
        base_z = np.zeros(nn)
        for i in range(nn):
            tt = i / (nn-1)
            base_x[i] = (1-tt)*r0[0] + tt*r1[0]
            base_y[i] = (1-tt)*r0[1] + tt*r1[1]
            base_z[i] = (1-tt)*r0[2] + tt*r1[2]

        #生成cst曲线
        cst_x,cst_y = cst_curve(nn,coef,None,xn1,xn2)
        cst_y = cst_y * L0

        self.curve_x = base_x + D[0]*cst_y
        self.curve_y = base_y + D[1]*cst_y
        self.curve_z = base_z + D[2]*cst_y

def read_sec_file(filename,scale = 1):
    
    df = pd.read_excel(filename,header=0)
    data = df.values
    z = data[1:,1]
    y = data[1:,2]
    x = data[1:,3]

    x,y,z = order_curve3d(x,y,z)

    x = np.append(x,x[0])
    y = np.append(y,y[0])
    z = np.append(z,z[0])

    return x*scale,y*scale,z*scale

def order_curve3d(x,y,z,axis = 'z'):
    
    if axis == 'x':
        x1 = y
        x2 = z
        x3 = x
    if axis == 'y':
        x1 = x
        x2 = z
        x3 = y
    if axis == 'z':  
        x1 = x
        x2 = y
        x3 = z

    cen_x = np.mean(x1)
    cen_y = np.mean(x2)
    
    x1s = []
    x2s = []
    x3s = []
    for i in range(len(x1)):
        dx = x1[i] - cen_x
        dy = x2[i] - cen_y
        ang = np.arctan2(dy,dx)
        if ang < 0:
            ang += np.pi*2
        x1s.append([x1[i],ang])
        x2s.append([x2[i],ang])
        x3s.append([x3[i],ang])

    x1s = sorted(x1s, key = lambda x:x[1])
    x2s = sorted(x2s, key = lambda x:x[1])
    x3s = sorted(x3s, key = lambda x:x[1])

    x1 = np.array([x[0] for x in x1s])
    x2 = np.array([x[0] for x in x2s])
    x3 = np.array([x[0] for x in x3s])

    if axis == 'x':
        return x3,x1,x2
    if axis == 'y':
        return x1,x3,x2
    if axis == 'z':  
        return x1,x2,x3

#%%

if __name__== "__main__":
    
    # Upstream duct
    AR = 5
    A_out = 5
    r_out = 1
    delta_x = 1
    delta_r = 0.5
    # HX
    l_hx = 1.5
    r_hx_out = 1.0
    r_hx_in = 0.8
    # Gaussian bump
    h_bump = r_hx_in
    # Downstream duct
    l_down_duct = 1  # 0.5
    # Fan
    dz_hx_fan = 0.0
    d_fan = 0.5
    l_fan = 0.3
    # Nozzle
    d_nozzle = 0.3
    l_nozzle = 0.3
    
    x_centroid_hx = 0.0
    y_centroid_hx = 0.0
    z_centroid_hx = 0.0
    
    outer_surface = BasicSurface(n_sec=55, name='out', nn=201, ns=51, projection = False)
    inner_surface = BasicSurface(n_sec=50+25, name='out', nn=201, ns=51, projection = False)
    
    ### NILS: upstream duct
    
    # =============================================================================
    (r_fun, z_fun, b_fun, phi_fun, z_hub_fun, r_hub_fun,
            z_casing_fun, r_casing_fun, r_in, b_in, b_out, A_in, m_target) = \
        parametrize_diffuser_v1(AR, A_out, r_out, delta_x, delta_r)
        
    m = np.array([0.0000000000, 0.0112950263, 0.0225900526, 0.0338850788, 0.0451801051, 0.0470907314, 0.0490013577, 0.0509119839, 0.0528226102, 0.0543322048, 0.0558417994, 0.0573513940, 0.0588609886, 0.0603705832, 0.0618801778, 0.0633897724, 0.0648993670, 0.0694162960, 0.0739332250, 0.0784501539, 0.0829670829, 0.0865509184, 0.0901347540, 0.0937185896, 0.0973024251, 0.0990594280, 0.1008164308, 0.1025734336, 0.1043304365, 0.1053236715, 0.1063169065, 0.1073101415, 0.1083033765, 0.1092966115, 0.1102898465, 0.1112830815, 0.1122763165, 0.1131644264, 0.1140525363, 0.1149406462, 0.1158287561, 0.1202693057, 0.1247098553, 0.1291504049, 0.1335909545, 0.1344779687, 0.1353649828, 0.1362519970, 0.1371390112, 0.1380260253, 0.1389130395, 0.1398000537, 0.1406870678, 0.1415540556, 0.1424210434, 0.1432880311, 0.1441550189, 0.1484899578, 0.1528248967, 0.1571598355, 0.1614947744, 0.1649697928, 0.1684448112, 0.1719198295, 0.1753948479, 0.1788698663, 0.1823448846, 0.1858199030, 0.1892949214, 0.1932543014, 0.1972136815, 0.2011730616, 0.2051324416, 0.2067702210, 0.2084080003, 0.2100457797, 0.2116835590, 0.2126121602, 0.2135407614, 0.2144693626, 0.2153979638, 0.2163265650, 0.2172551662, 0.2181837674, 0.2191123686, 0.2199446044, 0.2207768401, 0.2216090759, 0.2224413116, 0.2266024903, 0.2307636690, 0.2349248477, 0.2390860264, 0.2399234932, 0.2407609599, 0.2415984266, 0.2424358933, 0.2432733601, 0.2441108268, 0.2449482935, 0.2457857602, 0.2465503760, 0.2473149917, 0.2480796074, 0.2488442231, 0.2526673016, 0.2564903802, 0.2603134587, 0.2641365373, 0.2652027176, 0.2662688980, 0.2673350783, 0.2684012587, 0.2694674390, 0.2705336194, 0.2715997997, 0.2726659801, 0.2737735674, 0.2748811548, 0.2759887421, 0.2770963295, 0.2792277274, 0.2813591253, 0.2834905232, 0.2856219212, 0.2877533191, 0.2898847170, 0.2920161149, 0.2941475129, 0.2973809773, 0.3006144416, 0.3038479060, 0.3070813704, 0.3086701768, 0.3102589831, 0.3118477894, 0.3134365957, 0.3150254021, 0.3166142084, 0.3182030147, 0.3197918210, 0.3223372222, 0.3248826234, 0.3274280247, 0.3299734259, 0.3325188271, 0.3350642283, 0.3376096295, 0.3401550307, 0.3410866708, 0.3420183109, 0.3429499509, 0.3438815910, 0.3448132311, 0.3457448712, 0.3466765113, 0.3476081513, 0.3485976675, 0.3495871837, 0.3505766998, 0.3515662160, 0.3537959989, 0.3560257818, 0.3582555648, 0.3604853477, 0.3627151307, 0.3649449136, 0.3671746965, 0.3694044795, 0.3727519898, 0.3760995001, 0.3794470104, 0.3827945208, 0.3842111050, 0.3856276893, 0.3870442735, 0.3884608577, 0.3892784793, 0.3900961009, 0.3909137226, 0.3917313442, 0.3925489658, 0.3933665874, 0.3941842090, 0.3950018306, 0.3957450676, 0.3964883045, 0.3972315415, 0.3979747785, 0.4016909633, 0.4054071481, 0.4091233330, 0.4128395178, 0.4138802924, 0.4149210670, 0.4159618415, 0.4170026161, 0.4180433907, 0.4190841652, 0.4201249398, 0.4211657144, 0.4222437900, 0.4233218657, 0.4243999414, 0.4254780170, 0.4288405457, 0.4322030744, 0.4355656031, 0.4389281318, 0.4422906604, 0.4456531891, 0.4490157178, 0.4523782465, 0.4551378432, 0.4578974398, 0.4606570365, 0.4634166332, 0.4656456418, 0.4678746505, 0.4701036591, 0.4723326678, 0.4745616764, 0.4767906851, 0.4790196937, 0.4812487023, 0.4829167016, 0.4845847009, 0.4862527001, 0.4879206994, 0.4895886986, 0.4912566979, 0.4929246971, 0.4945926964, 0.4961639523, 0.4977352082, 0.4993064642, 0.5008777201, 0.5031626035, 0.5054474869, 0.5077323703, 0.5100172537, 0.5123021371, 0.5145870205, 0.5168719039, 0.5191567873, 0.5211785365, 0.5232002856, 0.5252220348, 0.5272437840, 0.5287841108, 0.5303244377, 0.5318647645, 0.5334050913, 0.5349454181, 0.5364857450, 0.5380260718, 0.5395663986, 0.5410396902, 0.5425129817, 0.5439862732, 0.5454595647, 0.5476281794, 0.5497967941, 0.5519654088, 0.5541340234, 0.5563026381, 0.5584712528, 0.5606398675, 0.5628084822, 0.5647461986, 0.5666839150, 0.5686216315, 0.5705593479, 0.5720470115, 0.5735346750, 0.5750223386, 0.5765100021, 0.5779976657, 0.5794853292, 0.5809729928, 0.5824606563, 0.5838925017, 0.5853243472, 0.5867561927, 0.5881880381, 0.5903069812, 0.5924259244, 0.5945448675, 0.5966638107, 0.5974587672, 0.5982537238, 0.5990486803, 0.5998436368, 0.6006385934, 0.6014335499, 0.6022285065, 0.6030234630, 0.6038903287, 0.6047571945, 0.6056240602, 0.6064909259, 0.6082290510, 0.6099671761, 0.6117053012, 0.6134434263, 0.6151815514, 0.6169196764, 0.6186578015, 0.6203959266, 0.6231278536, 0.6258597807, 0.6285917077, 0.6313236348, 0.6340555618, 0.6367874889, 0.6395194159, 0.6422513430, 0.6454999728, 0.6487486027, 0.6519972325, 0.6552458624, 0.6563654668, 0.6574850712, 0.6586046756, 0.6597242800, 0.6603115929, 0.6608989058, 0.6614862187, 0.6620735315, 0.6626608444, 0.6632481573, 0.6638354702, 0.6644227830, 0.6649913010, 0.6655598189, 0.6661283368, 0.6666968547, 0.6695394442, 0.6723820338, 0.6752246233, 0.6780672129, 0.6792146660, 0.6803621192, 0.6815095724, 0.6826570256, 0.6838044788, 0.6849519319, 0.6860993851, 0.6872468383, 0.6882197462, 0.6891926541, 0.6901655620, 0.6911384699, 0.6942365954, 0.6973347208, 0.7004328463, 0.7035309718, 0.7066290973, 0.7097272227, 0.7128253482, 0.7159234737, 0.7170735579, 0.7182236421, 0.7193737263, 0.7205238105, 0.7212143310, 0.7219048514, 0.7225953719, 0.7232858924, 0.7239764128, 0.7246669333, 0.7253574538, 0.7260479742, 0.7266958001, 0.7273436259, 0.7279914517, 0.7286392775, 0.7318784067, 0.7351175358, 0.7383566649, 0.7415957941, 0.7429728665, 0.7443499389, 0.7457270113, 0.7471040836, 0.7484811560, 0.7498582284, 0.7512353008, 0.7526123732, 0.7546726883, 0.7567330034, 0.7587933184, 0.7608536335, 0.7616326679, 0.7624117023, 0.7631907367, 0.7639697712, 0.7647488056, 0.7655278400, 0.7663068744, 0.7670859088, 0.7679405379, 0.7687951670, 0.7696497961, 0.7705044252, 0.7732984796, 0.7760925339, 0.7788865882, 0.7816806426, 0.7844746969, 0.7872687513, 0.7900628056, 0.7928568599, 0.7952405966, 0.7976243332, 0.8000080698, 0.8023918064, 0.8043902204, 0.8063886343, 0.8083870482, 0.8103854621, 0.8123838761, 0.8143822900, 0.8163807039, 0.8183791178, 0.8214627749, 0.8245464321, 0.8276300892, 0.8307137463, 0.8328669666, 0.8350201868, 0.8371734071, 0.8393266273, 0.8414798476, 0.8436330679, 0.8457862881, 0.8479395084, 0.8498756651, 0.8518118219, 0.8537479787, 0.8556841354, 0.8586937298, 0.8617033242, 0.8647129186, 0.8677225130, 0.8698366587, 0.8719508043, 0.8740649500, 0.8761790957, 0.8782932413, 0.8804073870, 0.8825215327, 0.8846356784, 0.8865460056, 0.8884563328, 0.8903666600, 0.8922769872, 0.8959931981, 0.8997094089, 0.9034256197, 0.9071418306, 0.9084002751, 0.9096587195, 0.9109171640, 0.9121756085, 0.9134340530, 0.9146924974, 0.9159509419, 0.9172093864, 0.9184702024, 0.9197310185, 0.9209918346, 0.9222526506, 0.9236420499, 0.9250314491, 0.9264208484, 0.9278102476, 0.9291996469, 0.9305890461, 0.9319784454, 0.9333678446, 0.9345091190, 0.9356503934, 0.9367916678, 0.9379329422, 0.9397144084, 0.9414958745, 0.9432773406, 0.9450588068, 0.9468402729, 0.9486217390, 0.9504032052, 0.9521846713, 0.9538520329, 0.9555193944, 0.9571867559, 0.9588541175, 0.9595214903, 0.9601888632, 0.9608562360, 0.9615236089, 0.9621909817, 0.9628583545, 0.9635257274, 0.9641931002, 0.9648271369, 0.9654611736, 0.9660952104, 0.9667292471, 0.9698994306, 0.9730696142, 0.9762397978, 0.9794099814, 0.9832952562, 0.9871805310, 0.9910658057, 0.9949510805, 0.9988363553, 1.0027216301, 1.0066069049, 1.0104921797, 1.0136235627, 1.0167549457, 1.0198863287, 1.0230177117, 1.0239730578, 1.0249284038, 1.0258837499, 1.0268390960, 1.0277944420, 1.0287497881, 1.0297051342, 1.0306604802, 1.0315081719, 1.0323558635, 1.0332035552, 1.0340512468, 1.0354533109, 1.0368553751, 1.0382574392, 1.0396595034, 1.0410615675, 1.0424636317, 1.0438656958, 1.0452677600, 1.0464197875, 1.0475718150, 1.0487238425, 1.0498758700, 1.0516782882, 1.0534807064, 1.0552831246, 1.0570855428, 1.0584988810, 1.0599122192, 1.0613255574, 1.0627388956, 1.0641522339, 1.0655655721, 1.0669789103, 1.0683922485, 1.0695582949, 1.0707243413, 1.0718903878, 1.0730564342, 1.0742224806, 1.0753885270, 1.0765545734, 1.0777206199, 1.0789123812, 1.0801041425, 1.0812959039, 1.0824876652, 1.0838253126, 1.0851629601, 1.0865006075, 1.0878382550, 1.0891759024, 1.0905135499, 1.0918511973, 1.0931888448, 1.0945203944, 1.0958519441, 1.0971834937, 1.0985150433, 1.0996280228, 1.1007410023, 1.1018539818, 1.1029669613, 1.1040799407, 1.1051929202, 1.1063058997, 1.1074188792, 1.1085688835, 1.1097188878, 1.1108688920, 1.1120188963, 1.1133207795, 1.1146226626, 1.1159245458, 1.1172264290, 1.1185283121, 1.1198301953, 1.1211320784, 1.1224339616, 1.1237391757, 1.1250443898, 1.1263496039, 1.1276548181, 1.1287564601, 1.1298581021, 1.1309597441, 1.1320613862, 1.1331630282, 1.1342646702, 1.1353663122, 1.1364679543, 1.1376107547, 1.1387535552, 1.1398963557, 1.1410391562, 1.1421379756, 1.1432367951, 1.1443356145, 1.1454344340, 1.1465332535, 1.1476320729, 1.1487308924, 1.1498297118, 1.1509708112, 1.1521119107, 1.1532530101, 1.1543941095, 1.1554926771, 1.1565912446, 1.1576898122, 1.1587883797, 1.1598869473, 1.1609855149, 1.1620840824, 1.1631826500, 1.1641377141, 1.1650927783, 1.1660478424, 1.1670029065, 1.1685639896, 1.1701250728, 1.1716861559, 1.1732472390, 1.1748083221, 1.1763694052, 1.1779304883, 1.1794915715, 1.1810083012, 1.1825250309, 1.1840417606, 1.1855584902, 1.1887026596, 1.1918468289, 1.1949909983, 1.1981351676, 1.2012793370, 1.2044235063, 1.2075676756, 1.2107118450, 1.2133810367, 1.2160502285, 1.2187194202, 1.2213886120, 1.2222718932, 1.2231551744, 1.2240384557, 1.2249217369, 1.2258050182, 1.2266882994, 1.2275715807, 1.2284548619, 1.2294204272, 1.2303859925, 1.2313515578, 1.2323171231, 1.2334031963, 1.2344892694, 1.2355753426, 1.2366614157, 1.2377474889, 1.2388335620, 1.2399196352, 1.2410057083, 1.2419604807, 1.2429152532, 1.2438700256, 1.2448247980, 1.2459038357, 1.2469828734, 1.2480619112, 1.2491409489, 1.2502199866, 1.2512990243, 1.2523780620, 1.2534570997, 1.2544091058, 1.2553611119, 1.2563131180, 1.2572651241, 1.2584023276, 1.2595395310, 1.2606767345, 1.2618139379, 1.2629511414, 1.2640883448, 1.2652255482, 1.2663627517, 1.2675530858, 1.2687434200, 1.2699337541, 1.2711240883, 1.2716477144, 1.2721713405, 1.2726949666, 1.2732185928, 1.2737422189, 1.2742658450, 1.2747894712, 1.2753130973, 1.2758485913, 1.2763840854, 1.2769195794, 1.2774550735, 1.2786798613, 1.2799046492, 1.2811294371, 1.2823542249, 1.2830082132, 1.2836622015, 1.2843161897, 1.2849701780, 1.2856241663, 1.2862781545, 1.2869321428, 1.2875861311, 1.2882280594, 1.2888699878, 1.2895119162, 1.2901538446, 1.2913067076, 1.2924595707, 1.2936124338, 1.2947652969, 1.2959181600, 1.2970710231, 1.2982238862, 1.2993767493, 1.3003903372, 1.3014039251, 1.3024175129, 1.3034311008, 1.3044587267, 1.3054863527, 1.3065139786, 1.3075416046, 1.3085692305, 1.3095968565, 1.3106244824, 1.3116521084, 1.3125809321, 1.3135097558, 1.3144385796, 1.3153674033, 1.3172902202, 1.3192130371, 1.3211358540, 1.3230586709, 1.3249814878, 1.3269043047, 1.3288271216, 1.3307499385, 1.3325743862, 1.3343988339, 1.3362232816, 1.3380477293, 1.3398721771, 1.3416966248, 1.3435210725, 1.3453455202, 1.3468356294, 1.3483257387, 1.3498158480, 1.3513059573, 1.3534448314, 1.3555837054, 1.3577225794, 1.3598614535, 1.3620003275, 1.3641392015, 1.3662780756, 1.3684169496, 1.3694411666, 1.3704653837, 1.3714896008, 1.3725138178, 1.3735380349, 1.3745622519, 1.3755864690, 1.3766106860, 1.3775668129, 1.3785229398, 1.3794790667, 1.3804351936, 1.3822793520, 1.3841235105, 1.3859676689, 1.3878118274, 1.3896559858, 1.3915001443, 1.3933443027, 1.3951884611, 1.3988147034, 1.4024409456, 1.4060671879, 1.4096934301, 1.4110614580, 1.4124294860, 1.4137975139, 1.4151655418, 1.4165335698, 1.4179015977, 1.4192696256, 1.4206376536, 1.4214721396, 1.4223066257, 1.4231411118, 1.4239755978, 1.4248100839, 1.4256445700, 1.4264790561, 1.4273135421, 1.4281626203, 1.4290116985, 1.4298607767, 1.4307098549, 1.4326727651, 1.4346356753, 1.4365985855, 1.4385614957, 1.4396256687, 1.4406898417, 1.4417540147, 1.4428181878, 1.4438823608, 1.4449465338, 1.4460107068, 1.4470748799, 1.4481317732, 1.4491886666, 1.4502455599, 1.4513024532, 1.4565869200, 1.4618713867, 1.4671558534, 1.4724403202, 1.4768473229, 1.4812543256, 1.4856613283, 1.4900683311, 1.4921253766, 1.4941824222, 1.4962394678, 1.4982965134, 1.5003535589, 1.5024106045, 1.5044676501, 1.5065246957, 1.5085166202, 1.5105085448, 1.5125004694, 1.5144923940, 1.5159343727, 1.5173763514, 1.5188183301, 1.5202603088, 1.5217022875, 1.5231442662, 1.5245862449, 1.5260282237, 1.5279116590, 1.5297950944, 1.5316785298, 1.5335619652, 1.5351055154, 1.5366490656, 1.5381926158, 1.5397361660, 1.5412797161, 1.5428232663, 1.5443668165, 1.5459103667, 1.5480221926, 1.5501340185, 1.5522458444, 1.5543576703, 1.5567082152, 1.5590587600, 1.5614093049, 1.5637598498, 1.5664205163, 1.5690811828, 1.5717418493, 1.5744025158, 1.5774776208, 1.5805527258, 1.5836278308, 1.5867029357, 1.5903531459, 1.5940033560, 1.5976535661, 1.6013037762, 1.6042445472, 1.6071853182, 1.6101260892, 1.6130668602, 1.6170275143, 1.6209881685, 1.6249488226, 1.6289094767])
    
    r = r_fun(m)
    z = z_fun(m)
    
    # =============================================================================
    # A_out, A_in, r_in, b_in, r_inner_in, r_outer_in, L_axial, L_m, Vol = ANNULARDIFFUSERMODEL._calculate_geometry(
    #     AR, r_out, b_out, delta_r, delta_x
    # )
    # =============================================================================
    
    # % Define the inner and outer surface vectors
    z_outer = z_casing_fun(m)
    z_inner = z_hub_fun(m)
    r_outer = r_casing_fun(m)
    r_inner = r_hub_fun(m)
    
    # =============================================================================
    l_hx = z_outer[-1] - z_inner[-1]
    
    z_outer = z_outer - delta_x - l_hx/2
    z_inner = z_inner - delta_x - l_hx/2
    # =============================================================================
    
    from scipy.interpolate import interp1d
    
    r_outer_interp = interp1d(z_outer, r_outer)
    r_inner_interp = interp1d(z_inner, r_inner)
    
    N_up_duct_slices = 50
    z_outer_slices = np.linspace(min(z_outer), max(z_outer), N_up_duct_slices)
    z_inner_slices = np.linspace(min(z_inner), max(z_inner), N_up_duct_slices)
    r_outer_slices = r_outer_interp(z_outer_slices)
    r_inner_slices = r_inner_interp(z_inner_slices)
    
    x_centroid_intake = x_centroid_hx - l_hx/2 - delta_x
    y_centroid_intake = y_centroid_hx
    z_centroid_intake = z_centroid_hx
    
    # Outer upstream duct surface
    
    for i in range(N_up_duct_slices):
    
        outer_surface.secs[i] = Section(n_curve = 4)
        outer_surface.secs[i].circle(
            np.array([y_centroid_intake - r_outer_slices[i], z_centroid_intake]),
            np.array([y_centroid_intake + r_outer_slices[i], z_centroid_intake]),
            np.array([y_centroid_intake, z_centroid_intake + r_outer_slices[i]]),
            nn=500,
        )
        outer_surface.secs[i].z = z_outer_slices[i] * np.ones_like(outer_surface.secs[i].x)
        #将曲线排序
        outer_surface.secs[i].x, outer_surface.secs[i].y, outer_surface.secs[i].z = order_curve3d(
            outer_surface.secs[i].x,
            outer_surface.secs[i].y,
            outer_surface.secs[i].z,
        )
        #让曲线封闭
        outer_surface.secs[i].x[0] = outer_surface.secs[i].x[-1]
        outer_surface.secs[i].y[0] = outer_surface.secs[i].y[-1]
        outer_surface.secs[i].z[0] = outer_surface.secs[i].z[-1]
    
    # =============================================================================
    # Inner upstream duct surface
    
    for i in range(N_up_duct_slices):
    
        inner_surface.secs[i] = Section(n_curve = 4)
        inner_surface.secs[i].circle(
            np.array([y_centroid_intake - r_inner_slices[i], z_centroid_intake]),
            np.array([y_centroid_intake + r_inner_slices[i], z_centroid_intake]),
            np.array([y_centroid_intake, z_centroid_intake + r_inner_slices[i]]),
            nn=500,
        )
        inner_surface.secs[i].z = z_inner_slices[i] * np.ones_like(inner_surface.secs[i].x)
        #将曲线排序
        inner_surface.secs[i].x, inner_surface.secs[i].y, inner_surface.secs[i].z = order_curve3d(
            inner_surface.secs[i].x,
            inner_surface.secs[i].y,
            inner_surface.secs[i].z,
        )
        #让曲线封闭
        inner_surface.secs[i].x[0] = inner_surface.secs[i].x[-1]
        inner_surface.secs[i].y[0] = inner_surface.secs[i].y[-1]
        inner_surface.secs[i].z[0] = inner_surface.secs[i].z[-1]
    # =============================================================================
    # =============================================================================
    
    i_up_duct_out = N_up_duct_slices - 1
    
    # =============================================================================
    x_centroid_bump = x_centroid_hx - l_hx/2
    y_centroid_bump = y_centroid_hx
    z_centroid_bump = z_centroid_hx
    
    x_base = np.linspace(0, 1, 50)  # np.linspace(-r_hx_in, r_hx_in, 50)
    # y_base = np.zeros_like(x_base)
    xc_bump = 0.5
    w_bump = 2 * r_hx_in
    y_bump = bump_function(x_base, xc_bump, h_bump, w_bump, 'G')
    x_bump = -r_hx_in + x_base * 2 * r_hx_in
    
    # fig, ax = plt.subplots()
    # ax.plot(x_bump, y_bump, marker='.')
    # ax.set_aspect('equal')
    # plt.show()
    
    N_bump_slices = len(x_base)//2
    
    x_bump_slices = y_bump[:len(x_base)//2] - l_hx/2
    r_bump_slices = x_bump[:len(x_base)//2]
    
    for i in range(N_bump_slices):
        
        inner_surface.secs[i + N_up_duct_slices] = Section(n_curve = 4)
        inner_surface.secs[i + N_up_duct_slices].circle(
            np.array([y_centroid_bump - r_bump_slices[i], z_centroid_bump]),
            np.array([y_centroid_bump + r_bump_slices[i], z_centroid_bump]),
            np.array([y_centroid_bump, z_centroid_bump + r_bump_slices[i]]),
            nn=500,
        )
        inner_surface.secs[i + N_up_duct_slices].z = x_bump_slices[i] * np.ones_like(inner_surface.secs[i + N_up_duct_slices].x)
        #将曲线排序
        inner_surface.secs[i + N_up_duct_slices].x, inner_surface.secs[i + N_up_duct_slices].y, inner_surface.secs[i + N_up_duct_slices].z = order_curve3d(
            inner_surface.secs[i + N_up_duct_slices].x,
            inner_surface.secs[i + N_up_duct_slices].y,
            inner_surface.secs[i + N_up_duct_slices].z,
        )
        #让曲线封闭
        inner_surface.secs[i + N_up_duct_slices].x[0] = inner_surface.secs[i + N_up_duct_slices].x[-1]
        inner_surface.secs[i + N_up_duct_slices].y[0] = inner_surface.secs[i + N_up_duct_slices].y[-1]
        inner_surface.secs[i + N_up_duct_slices].z[0] = inner_surface.secs[i + N_up_duct_slices].z[-1]
    
    # inner_surface.geo(update_sec = False)
    
    # inner_surface.smooth(0,N_up_duct_slices-1,smooth0=True,ratio_end=-1)
    # inner_surface.smooth(N_up_duct_slices,N_up_duct_slices+N_bump_slices-1,smooth0=True,ratio_end=-1)

    # inner_surface.flip(axis = '+Y')
    # inner_surface.flip(axis = '+X')

    # inner_surface.output_tecplot(fname='engine_inner_surface_new.dat')  # NILS
    
    # import pyvista as pv
    
    # mesh = pv.read('engine_inner_surface_new.dat')
    # cpos = mesh.plot(border=True, border_color='k')
    
    # sys.exit()
    # =============================================================================
    
    # ### NILS: inner hx forward limit
    
    # x_centroid_hx_in_fwd = x_centroid_hx - l_hx/2
    # y_centroid_hx_in_fwd = y_centroid_hx
    # z_centroid_hx_in_fwd = z_centroid_hx
    
    # outer_surface.secs[1 + i_up_duct_out] = Section(n_curve = 4)
    # outer_surface.secs[1 + i_up_duct_out].circle(
    #     np.array([y_centroid_hx_in_fwd - r_hx_in, z_centroid_hx_in_fwd]),
    #     np.array([y_centroid_hx_in_fwd + r_hx_in, z_centroid_hx_in_fwd]),
    #     np.array([y_centroid_hx_in_fwd, z_centroid_hx_in_fwd + r_hx_in]),
    #     nn=500,
    # )
    # outer_surface.secs[1 + i_up_duct_out].z = x_centroid_hx_in_fwd * np.ones_like(outer_surface.secs[1 + i_up_duct_out].x)
    # #将曲线排序
    # outer_surface.secs[1 + i_up_duct_out].x, outer_surface.secs[1 + i_up_duct_out].y, outer_surface.secs[1 + i_up_duct_out].z = order_curve3d(
    #     outer_surface.secs[1 + i_up_duct_out].x,
    #     outer_surface.secs[1 + i_up_duct_out].y,
    #     outer_surface.secs[1 + i_up_duct_out].z,
    # )
    # #让曲线封闭
    # outer_surface.secs[1 + i_up_duct_out].x[0] = outer_surface.secs[1 + i_up_duct_out].x[-1]
    # outer_surface.secs[1 + i_up_duct_out].y[0] = outer_surface.secs[1 + i_up_duct_out].y[-1]
    # outer_surface.secs[1 + i_up_duct_out].z[0] = outer_surface.secs[1 + i_up_duct_out].z[-1]
    
    ### NILS: inner hx aft limit
    
    x_centroid_hx_in_aft = x_centroid_hx + l_hx/2
    y_centroid_hx_in_aft = y_centroid_hx
    z_centroid_hx_in_aft = z_centroid_hx
    
    outer_surface.secs[1 + i_up_duct_out] = Section(n_curve = 4)
    outer_surface.secs[1 + i_up_duct_out].circle(
        np.array([y_centroid_hx_in_aft - r_hx_in, z_centroid_hx_in_aft]),
        np.array([y_centroid_hx_in_aft + r_hx_in, z_centroid_hx_in_aft]),
        np.array([y_centroid_hx_in_aft, z_centroid_hx_in_aft + r_hx_in]),
        nn=500,
    )
    outer_surface.secs[1 + i_up_duct_out].z = x_centroid_hx_in_aft * np.ones_like(outer_surface.secs[1 + i_up_duct_out].x)
    #将曲线排序
    outer_surface.secs[1 + i_up_duct_out].x, outer_surface.secs[1 + i_up_duct_out].y, outer_surface.secs[1 + i_up_duct_out].z = order_curve3d(
        outer_surface.secs[1 + i_up_duct_out].x,
        outer_surface.secs[1 + i_up_duct_out].y,
        outer_surface.secs[1 + i_up_duct_out].z,
    )
    #让曲线封闭
    outer_surface.secs[1 + i_up_duct_out].x[0] = outer_surface.secs[1 + i_up_duct_out].x[-1]
    outer_surface.secs[1 + i_up_duct_out].y[0] = outer_surface.secs[1 + i_up_duct_out].y[-1]
    outer_surface.secs[1 + i_up_duct_out].z[0] = outer_surface.secs[1 + i_up_duct_out].z[-1]
    
    # =============================================================================
    ### NILS: intermediate surface to facilitate smoothing of downstream duct
    
    x_centroid_hx_in_aft = x_centroid_hx + l_hx/2 + l_down_duct/2
    y_centroid_hx_in_aft = y_centroid_hx
    z_centroid_hx_in_aft = z_centroid_hx
    
    outer_surface.secs[2 + i_up_duct_out] = Section(n_curve = 4)
    outer_surface.secs[2 + i_up_duct_out].circle(
        np.array([y_centroid_hx_in_aft - r_hx_in * 0.5, z_centroid_hx_in_aft]),
        np.array([y_centroid_hx_in_aft + r_hx_in * 0.5, z_centroid_hx_in_aft]),
        np.array([y_centroid_hx_in_aft, z_centroid_hx_in_aft + r_hx_in * 0.5]),
        nn=500,
    )
    outer_surface.secs[2 + i_up_duct_out].z = x_centroid_hx_in_aft * np.ones_like(outer_surface.secs[2 + i_up_duct_out].x)
    #将曲线排序
    outer_surface.secs[2 + i_up_duct_out].x, outer_surface.secs[2 + i_up_duct_out].y, outer_surface.secs[2 + i_up_duct_out].z = order_curve3d(
        outer_surface.secs[2 + i_up_duct_out].x,
        outer_surface.secs[2 + i_up_duct_out].y,
        outer_surface.secs[2 + i_up_duct_out].z,
    )
    #让曲线封闭
    outer_surface.secs[2 + i_up_duct_out].x[0] = outer_surface.secs[2 + i_up_duct_out].x[-1]
    outer_surface.secs[2 + i_up_duct_out].y[0] = outer_surface.secs[2 + i_up_duct_out].y[-1]
    outer_surface.secs[2 + i_up_duct_out].z[0] = outer_surface.secs[2 + i_up_duct_out].z[-1]
    # =============================================================================
    
    ### NILS: fan inlet
    
    x_centroid_fan_inlet = x_centroid_hx + l_hx/2 + l_down_duct
    y_centroid_fan_inlet = z_centroid_hx  # NILS: note coordinate change
    z_centroid_fan_inlet = y_centroid_hx
    
    outer_surface.secs[3 + i_up_duct_out] = Section(n_curve = 4)
    outer_surface.secs[3 + i_up_duct_out].circle(
        np.array([y_centroid_fan_inlet - d_fan/2, z_centroid_fan_inlet]),
        np.array([y_centroid_fan_inlet + d_fan/2, z_centroid_fan_inlet]),
        np.array([y_centroid_fan_inlet, z_centroid_fan_inlet + d_fan/2]),
        nn=500,
    )
    outer_surface.secs[3 + i_up_duct_out].z = x_centroid_fan_inlet * np.ones_like(outer_surface.secs[3 + i_up_duct_out].x)
    #将曲线排序
    outer_surface.secs[3 + i_up_duct_out].x, outer_surface.secs[3 + i_up_duct_out].y, outer_surface.secs[3 + i_up_duct_out].z = order_curve3d(
        outer_surface.secs[3 + i_up_duct_out].x,
        outer_surface.secs[3 + i_up_duct_out].y,
        outer_surface.secs[3 + i_up_duct_out].z,
    )
    #让曲线封闭
    outer_surface.secs[3 + i_up_duct_out].x[0] = outer_surface.secs[3 + i_up_duct_out].x[-1]
    outer_surface.secs[3 + i_up_duct_out].y[0] = outer_surface.secs[3 + i_up_duct_out].y[-1]
    outer_surface.secs[3 + i_up_duct_out].z[0] = outer_surface.secs[3 + i_up_duct_out].z[-1]
    
    ### NILS: fan outlet
    
    x_centroid_fan_outlet = x_centroid_hx + l_hx/2 + l_down_duct + l_fan
    y_centroid_fan_outlet = y_centroid_hx + dz_hx_fan  # NILS: note coordinate change
    z_centroid_fan_outlet = z_centroid_hx
    
    outer_surface.secs[4 + i_up_duct_out] = Section(n_curve = 4)
    outer_surface.secs[4 + i_up_duct_out].circle(
        np.array([y_centroid_fan_outlet - d_fan/2, z_centroid_fan_outlet]),
        np.array([y_centroid_fan_outlet + d_fan/2, z_centroid_fan_outlet]),
        np.array([y_centroid_fan_outlet, z_centroid_fan_outlet + d_fan/2]),
        nn=500,
    )
    outer_surface.secs[4 + i_up_duct_out].z = x_centroid_fan_outlet * np.ones_like(outer_surface.secs[4 + i_up_duct_out].x)
    #将曲线排序
    outer_surface.secs[4 + i_up_duct_out].x, outer_surface.secs[4 + i_up_duct_out].y, outer_surface.secs[4 + i_up_duct_out].z = order_curve3d(
        outer_surface.secs[4 + i_up_duct_out].x,
        outer_surface.secs[4 + i_up_duct_out].y,
        outer_surface.secs[4 + i_up_duct_out].z,
    )
    #让曲线封闭
    outer_surface.secs[4 + i_up_duct_out].x[0] = outer_surface.secs[4 + i_up_duct_out].x[-1]
    outer_surface.secs[4 + i_up_duct_out].y[0] = outer_surface.secs[4 + i_up_duct_out].y[-1]
    outer_surface.secs[4 + i_up_duct_out].z[0] = outer_surface.secs[4 + i_up_duct_out].z[-1]
    
    ### NILS: nozzle outlet
    
    x_centroid_nozzle_outlet = x_centroid_hx + l_hx/2 + l_down_duct + l_fan + l_nozzle
    y_centroid_nozzle_outlet = y_centroid_hx + dz_hx_fan  # NILS: note coordinate change
    z_centroid_nozzle_outlet = z_centroid_hx
    
    outer_surface.secs[5 + i_up_duct_out] = Section(n_curve = 4)
    outer_surface.secs[5 + i_up_duct_out].circle(
        np.array([y_centroid_nozzle_outlet - d_nozzle/2, z_centroid_nozzle_outlet]),
        np.array([y_centroid_nozzle_outlet + d_nozzle/2, z_centroid_nozzle_outlet]),
        np.array([y_centroid_nozzle_outlet, z_centroid_nozzle_outlet + d_nozzle/2]),
        nn=500,
    )
    outer_surface.secs[5 + i_up_duct_out].z = x_centroid_nozzle_outlet * np.ones_like(outer_surface.secs[5 + i_up_duct_out].x)
    #将曲线排序
    outer_surface.secs[5 + i_up_duct_out].x, outer_surface.secs[5 + i_up_duct_out].y, outer_surface.secs[5 + i_up_duct_out].z = order_curve3d(
        outer_surface.secs[5 + i_up_duct_out].x,
        outer_surface.secs[5 + i_up_duct_out].y,
        outer_surface.secs[5 + i_up_duct_out].z,
    )
    #让曲线封闭
    outer_surface.secs[5 + i_up_duct_out].x[0] = outer_surface.secs[5 + i_up_duct_out].x[-1]
    outer_surface.secs[5 + i_up_duct_out].y[0] = outer_surface.secs[5 + i_up_duct_out].y[-1]
    outer_surface.secs[5 + i_up_duct_out].z[0] = outer_surface.secs[5 + i_up_duct_out].z[-1]
    
    # Plot sections
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for i in range(len(outer_surface.secs)):
    #     ax.scatter(outer_surface.secs[i].x, outer_surface.secs[i].y, outer_surface.secs[i].z)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_aspect('equal')
    # plt.show()
    
    # sys.exit()
    
    # Convert to mesh
    
    outer_surface.geo(update_sec = False)
    
    outer_surface.smooth(0,N_up_duct_slices-1,smooth0=True,ratio_end=-1)
    outer_surface.smooth(1 + i_up_duct_out, 3 + i_up_duct_out, smooth0=False, ratio_end=-1)  # NILS: important to set `smooth0=False`, else smoothing will go crazy

    outer_surface.flip(axis = '+Y')
    outer_surface.flip(axis = '+X')

    outer_surface.output_tecplot(fname='engine_outer_surface_new.dat')  # NILS
    
    # =============================================================================
    inner_surface.geo(update_sec = False)
    
    inner_surface.smooth(0,N_up_duct_slices-1,smooth0=True,ratio_end=-1)
    inner_surface.smooth(N_up_duct_slices,N_up_duct_slices+N_bump_slices-1,smooth0=True,ratio_end=-1)

    inner_surface.flip(axis = '+Y')
    inner_surface.flip(axis = '+X')

    inner_surface.output_tecplot(fname='engine_inner_surface_new.dat')  # NILS
    # =============================================================================
    
    #%%
    
    # import pyvista as pv
    
    # mesh = pv.read('engine_outer_surface_new.dat')
    # cpos = mesh.plot(border=True, border_color='k')
    
    # =============================================================================
    import pyvista as pv
    
    mesh1 = pv.read('engine_outer_surface_new.dat')
    mesh2 = pv.read('engine_inner_surface_new.dat')
    
    # Create a plotter
    plotter = pv.Plotter()
    
    # Add both meshes
    plotter.add_mesh(mesh1, color='lightblue')
    plotter.add_mesh(mesh2, color='red')
    
    # Optional border
    plotter.show_bounds(color='k')
    
    # Show the plot
    plotter.show()
    # =============================================================================
    
    

    
