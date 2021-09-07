import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import Ridge
from scipy.special import expit
from scipy.integrate import solve_ivp

class HillTypeMuscle:
    """
    Damped Hill-type muscle model adapted from Millard et al. (2013). The
    dynamic model is defined in terms of normalized length and velocity.
    To model a particular muscle, scale factors are needed for force, CE
    length, and SE length. These are given as constructor arguments. 
    """

    def __init__(self, f0M, resting_length_muscle, resting_length_tendon):
        """
        :param f0M: maximum isometric force
        :param resting_length_muscle: actual length (m) of muscle (CE) that corresponds to
            normalized length of 1
        :param resting_length_tendon: actual length of tendon (m) that corresponds to
            normalized length of 1
        """
        self.f0M = f0M
        self.resting_length_muscle = resting_length_muscle
        self.resting_length_tendon = resting_length_tendon

    def norm_tendon_length(self, muscle_tendon_length, normalized_muscle_length):
        """
        :param muscle_tendon_length: non-normalized length of the full muscle-tendon
            complex (typically found from joint angles and musculoskeletal geometry)
        :param normalized_muscle_length: normalized length of the contractile element
            (the state variable of the muscle model)
        :return: normalized length of the tendon
        """
        return (muscle_tendon_length - self.resting_length_muscle * normalized_muscle_length) / self.resting_length_tendon

    def get_force(self, total_length, norm_muscle_length):
        """
        :param total_length: muscle-tendon length (m)
        :param norm_muscle_length: normalized length of muscle (the state variable)
        :return: muscle tension (N)
        """
        return self.f0M * force_length_tendon(self.norm_tendon_length(total_length, norm_muscle_length))

def get_velocity(a, lm, lt):
    """
    :param a: activation (between 0 and 1)
    :param lm: normalized length of muscle (contractile element)
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized lengthening velocity of muscle (contractile element)
    """

    beta = 0.1 # damping coefficient (see damped model in Millard et al.)

    alpha = 0

    def f(vm):
        p1 = a * force_length_muscle(lm) * force_velocity_muscle(vm)
        p2 = force_length_parallel(lm) + (beta * vm * np.cos(alpha))
        p3 = force_length_tendon(lt)
        return p1 + p2 - p3

    return fsolve(f, 0)

def force_length_tendon(lt):
    """
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized tension produced by tendon
    """

    # slack length of SE
    slack_len_SE = 1.0
    tension_norm = []

    if isinstance(lt, float) or isinstance(lt, int):
        lt = np.array([lt])

    for i in range(len(lt)):
        if lt[i] < slack_len_SE:
            tension_norm.append(0)
        else:
            tension_norm.append((10.0 * (lt[i] - slack_len_SE)) + (240.0 * (lt[i] - slack_len_SE)**2))

    return np.array(tension_norm)

def force_length_parallel(lm):
    """
    :param lm: normalized length of muscle (contractile element)
    :return: normalized force produced by parallel elastic element
    """

    # slack length of PE
    slack_len_PE = 1.0
    force_norm = []

    if  isinstance(lm, float) or isinstance(lm, int):
        lm = np.array([lm])

    for i in range(len(lm)):
        if lm[i] < slack_len_PE:
            force_norm.append(0)
        else:
            force_norm.append((3.0 * (lm[i] - slack_len_PE)**2) / (.6 + lm[i] - slack_len_PE))

    return np.array((force_norm))

def plot_curves():
    """
    Plot force-length, force-velocity, SE, and PE curves.
    """
    lm = np.arange(0, 1.8, .01)
    vm = np.arange(-1.2, 1.2, .01)
    lt = np.arange(0, 1.07, .01)
    plt.figure(dpi=150)
    plt.subplot(2,1,1)
    plt.plot(lm, force_length_muscle(lm), 'r')
    plt.plot(lm, force_length_parallel(lm), 'g')
    plt.plot(lt, force_length_tendon(lt), 'b')
    plt.legend(('CE', 'PE', 'SE'))
    plt.xlabel('Normalized length')
    plt.ylabel('Force scale factor')
    plt.subplot(2, 1, 2)
    plt.plot(vm, force_velocity_muscle(vm), 'k')
    plt.xlabel('Normalized muscle velocity')
    plt.ylabel('Force scale factor')
    plt.tight_layout()
    plt.show()


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-(x-self.mu)**2/2/self.sigma**2)


class Sigmoid:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return expit((x-self.mu) / self.sigma)


class Regression():
    """
    1D regression model with Gaussian basis functions.
    """

    def __init__(self, x, t, centres, width, regularization_weight=1e-6, sigmoids=False):
        """
        :param x: samples of an independent variable
        :param t: corresponding samples of a dependent variable
        :param centres: a vector of Gaussian centres (should have similar range of values as x)
        :param width: sigma parameter of Gaussians
        :param regularization_weight: regularization strength parameter
        """
        if sigmoids:
            self.basis_functions = [Sigmoid(centre, width) for centre in centres]
        else:
            self.basis_functions = [Gaussian(centre, width) for centre in centres]
        self.ridge = Ridge(alpha=regularization_weight, fit_intercept=False)
        self.ridge.fit(self._get_features(x), t)

    def eval(self, x):
        """
        :param x: a new (or multiple samples) of the independent variable
        :return: the value of the curve at x
        """
        return self.ridge.predict(self._get_features(x))

    def _get_features(self, x):
        if not isinstance(x, collections.Sized):
            x = [x]

        phi = np.zeros((len(x), len(self.basis_functions)))
        for i, basis_function in enumerate(self.basis_functions):
            phi[:,i] = basis_function(x)
        return phi


def get_muscle_force_velocity_regression():
    data = np.array([
        [-1.0028395556708567, 0.0024834319945283845],
        [-0.8858611825192801, 0.03218792009622429],
        [-0.5176245843258415, 0.15771090304473967],
        [-0.5232565269687035, 0.16930496922242444],
        [-0.29749770052593094, 0.2899790099290114],
        [-0.2828848376217543, 0.3545364496120378],
        [-0.1801231103040022, 0.3892195938775034],
        [-0.08494610976156225, 0.5927831890757294],
        [-0.10185137142991896, 0.6259097662790973],
        [-0.0326643239546236, 0.7682365981934388],
        [-0.020787245583830716, 0.8526638522676352],
        [0.0028442725407418212, 0.9999952831301149],
        [0.014617579774061973, 1.0662107025777694],
        [0.04058866536166583, 1.124136223202283],
        [0.026390887007381902, 1.132426122025424],
        [0.021070257776939272, 1.1986556920827338],
        [0.05844673474682183, 1.2582274002971627],
        [0.09900238201929201, 1.3757434966156459],
        [0.1020023112662436, 1.4022310794556732],
        [0.10055894908138963, 1.1489210160137733],
        [0.1946227683309354, 1.1571212943090965],
        [0.3313459588217258, 1.152041225442796],
        [0.5510200231126625, 1.204839508502158]
    ])

    velocity = data[:,0]
    force = data[:,1]

    centres = np.arange(-1, 0, .2)
    width = .15
    result = Regression(velocity, force, centres, width, .1, sigmoids=True)

    return result


def get_muscle_force_length_regression():
    """
    CE force-length data samples from Winters et al. (2011) Figure 3C,
    normalized so that max force is ~1 and length at max force is ~1.
    The sampples were taken form the paper with WebPlotDigitizer, and
    cut-and-pasted here.

    WRITE CODE HERE 1) Use WebPlotDigitizer to extract force-length points
    from Winters et al. (2011) Figure 3C, which is on Learn. Click
    "View Data", select all, cut, and paste below. 2) Normalize the data
    so optimal length = 1 and peak = 1. 3) Return a Regression object that
    uses Gaussian basis functions. 
    """

    data = np.array([
        [15.1032448378, 3.71515227056],
        [18.8790560472, 1.75661639453],
        [11.9174041298, 9.90111762018],
        [18.4070796460, 14.7127009847],
        [18.2890855457, 16.1207362167],
        [16.6371681416, 18.0867505920],
        [13.5693215339, 15.2590469068],
        [14.7492625369, 24.5589762765],
        [16.6371681416, 21.7487224230],
        [20.6489675516, 23.7346794632],
        [21.4749262537, 23.7375877685],
        [22.1828908555, 22.3316298974],
        [18.2890855457, 27.1066517097],
        [18.2890855457, 32.1770742449],
        [19.2330383481, 32.4620881632],
        [16.6371681416, 36.9599900287],
        [21.3569321534, 35.0047779301],
        [20.1769911504, 42.0428767294],
        [21.4749262537, 44.8643483319],
        [23.0088495575, 45.7148198928],
        [25.0147492625, 43.7500519340],
        [24.8967551622, 46.5665378703],
        [26.7846607670, 44.8830445802],
        [20.5309734513, 46.5511653995],
        [20.6489675516, 49.0867921393],
        [21.0029498525, 50.4964892600],
        [21.5929203540, 53.8788483111],
        [21.9469026549, 57.2603764178],
        [24.5427728614, 53.3258548340],
        [24.4247787611, 54.1705097844],
        [23.1268436578, 60.9265029706],
        [25.3687315634, 67.9766504633],
        [26.3126843658, 63.4729319872],
        [27.9646017699, 62.6336781752],
        [28.3185840708, 67.1419668453],
        [30.2064896755, 63.2049524284],
        [27.8466076696, 71.6473472101],
        [26.4306784661, 71.9240516847],
        [25.3687315634, 70.7935518717],
        [26.3126843658, 73.8954671985],
        [26.6666666667, 76.1502347418],
        [27.2566371681, 80.6593543562],
        [28.3185840708, 81.7898541693],
        [28.9085545723, 83.2003822344],
        [30.4424778761, 81.7973326686],
        [30.4424778761, 81.7973326686],
        [31.6224188791, 81.8014873904],
        [31.7404129794, 85.4638746936],
        [31.2684365782, 86.8706635091],
        [30.2064896755, 85.1767834144],
        [32.5663716814, 87.7203041256],
        [33.1563421829, 87.1590012049],
        [33.1563421829, 85.1871702190],
        [33.1563421829, 90.8209730359],
        [34.3362831858, 89.6983671943],
        [34.5722713864, 91.3893389838],
        [33.3923303835, 80.6809589098],
        [33.2743362832, 78.7087124517],
        [33.8643067847, 78.7107898126],
        [33.0383480826, 74.7642195355],
        [31.2684365782, 76.1664381570],
        [37.6401179941, 79.0057750634],
        [37.6401179941, 84.0761975986],
        [37.1681415929, 89.4266483859],
        [36.2241887906, 89.1416344676],
        [38.3480825959, 92.2477045162],
        [37.7581120944, 92.2456271553],
        [38.5840707965, 93.9386763056],
        [38.9380530973, 94.5033030039],
        [37.5221238938, 94.7800074785],
        [37.6401179941, 96.7522539366],
        [38.2300884956, 96.7543312975],
        [39.5280235988, 99.5758029000],
        [41.0619469027, 96.2009223482],
        [41.8879056047, 96.2038306535],
        [42.2418879056, 99.8670489011],
        [42.8318584071, 99.8691262620],
        [43.7758112094, 97.6189289127],
        [44.1297935103, 100.437076738],
        [44.3657817109, 99.5928372595],
        [45.5457227139, 96.4984004321],
        [44.4837758112, 91.7059287881],
        [45.4277286136, 91.1458722839],
        [46.7256637168, 91.4321326187],
        [47.4336283186, 95.3782874237],
        [47.9056047198, 95.6616394532],
        [47.7876106195, 97.0696746853],
        [46.9616519174, 97.9118368025],
        [46.1356932153, 99.3173792015],
        [48.9675516224, 99.6090406747],
        [50.1474926254, 95.3878432839],
        [48.8495575221, 93.4114421040],
        [50.1474926254, 92.5709418754],
        [50.1474926254, 88.3455897628],
        [50.1474926254, 84.4019277909],
        [50.2654867257, 79.8953010096],
        [50.1474926254, 77.0779841290],
        [52.2713864307, 80.4657443184],
        [51.4454277286, 89.7586106610],
        [51.7994100295, 96.8021105987],
        [53.9233038348, 90.0490257177],
        [54.0412979351, 86.9508496406],
        [53.8053097345, 86.1049482737],
        [52.7433628319, 86.3828991649],
        [55.1032448378, 87.2362790311],
        [55.1032448378, 82.1658564959],
        [54.1592920354, 80.4723918734],
        [53.5693215339, 79.6252440899],
        [53.3333333333, 80.7511737089],
        [54.3952802360, 76.5295608459],
        [56.1651917404, 76.2541027878],
        [56.9911504425, 73.1584195438],
        [57.8171091445, 72.5979475674],
        [58.1710914454, 75.9794756741],
        [57.1091445428, 68.6517927625],
        [57.2271386431, 66.6803772487],
        [56.4011799410, 64.4239478167],
        [58.8790560472, 66.6861938593],
        [59.3510324484, 63.8709543396],
        [60.4129793510, 63.0296231667],
        [61.4749262537, 60.2164610079],
        [53.3333333333, 59.6244131455],
        [53.4513274336, 53.1459553783],
        [55.5752212389, 52.0266733142],
        [55.4572271386, 53.9980888280],
        [57.2271386431, 48.0888279530],
        [59.9410029499, 52.0420457850],
        [59.3510324484, 43.0258839171],
        [59.9410029499, 35.7040176160],
        [61.7109144543, 27.5412356143],
        [64.1887905605, 29.8034816569],
        [64.8967551622, 29.8059744900],
        [66.1946902655, 34.5992770784],
        [69.4985250737, 34.8926004404],
        [68.7905604720, 26.4394033819],
        [67.6106194690, 25.3084880967],
        [62.8908554572, 41.9115875192],
        [64.7787610619, 48.6787984544],
        [69.4985250737, 18.8362624122],
        [69.3805309735, 17.9907765175],
        [69.3805309735, 13.2020441231],
        [71.1504424779, 13.2082762059],
        [72.5663716814, 12.6498815904],
        [72.3303834808, 18.2828534630],
        [74.1002949853, 9.27500103868],
    ])

    length = data[:, 0]
    tension = data[:, 1]
    max_tension = max(tension)
    length_idx = np.argmax(tension)
    length_max_tension = length[length_idx]
    norm_length = length / length_max_tension
    norm_tension = tension / max_tension

    centres = np.arange(min(norm_length) + 0.1, max(norm_length), .2)
    width = .15
    result = Regression(norm_length, norm_tension, centres, width, .1, sigmoids=False)

    return result

force_length_regression = get_muscle_force_length_regression()
force_velocity_regression = get_muscle_force_velocity_regression()


def force_length_muscle(lm):
    """
    :param lm: muscle (contracile element) length
    :return: force-length scale factor
    """
    return force_length_regression.eval(lm)


def force_velocity_muscle(vm):
    """
    :param vm: muscle (contractile element) velocity)
    :return: force-velocity scale factor
    """
    return np.maximum(0, force_velocity_regression.eval(vm))


if __name__ == "__main__":

    # Q1
    plot_curves()

    # Q2
    print(get_velocity(1, 1, 1.01))

    # Q3
    f0M = 100.0
    resting_muscle_length = 0.3
    resting_tendon_length = 0.1

    total_length = resting_muscle_length + resting_tendon_length
    muscle = HillTypeMuscle(f0M, resting_muscle_length, resting_tendon_length)

    def find_lm(time, x):

        if time < 0.5:
            a = 0
        else:
            a = 1

        velocity_norm = get_velocity(a, x, muscle.norm_tendon_length(total_length,x))
        return velocity_norm

    sol = solve_ivp(find_lm, [0, 2], np.array([1.0]), max_step = 0.01)
    ce_force = muscle.get_force(total_length, sol.y.T)

    plt.figure(dpi=150)
    plt.subplot(2,1,1)
    plt.plot(sol.t, sol.y.T)
    plt.xlabel('Time (s)')
    plt.ylabel('CE Length (m)')
    plt.subplot(2,1,2)
    plt.plot(sol.t, ce_force, 'g')
    plt.xlabel('Time (s)')
    plt.ylabel('CE Force (N)')

    plt.tight_layout()
    plt.show()
