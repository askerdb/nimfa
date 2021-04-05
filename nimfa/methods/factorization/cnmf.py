
"""
###################################
C-Nmf (``methods.factorization.cnmf``)
###################################

"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *
import cvxpy as cvx

__all__ = ['Cnmf']


class Cnmf(nmf_std.Nmf_std):
    """
    :param V: The target matrix to estimate.
    :type V: Instance of the :class:`scipy.sparse` sparse matrices types,
       :class:`numpy.ndarray`, :class:`numpy.matrix` or tuple of instances of
       the latter classes.

    :param seed: Specify method to seed the computation of a factorization. If
       specified :param:`W` and :param:`H` seeding must be None. If neither seeding
       method or initial fixed factorization is specified, random initialization is
       used.
    :type seed: `str` naming the method or :class:`methods.seeding.nndsvd.Nndsvd`
       or None

    :param W: Specify initial factorization of basis matrix W. Default is None.
       When specified, :param:`seed` must be None.
    :type W: :class:`scipy.sparse` or :class:`numpy.ndarray` or
       :class:`numpy.matrix` or None

    :param H: Specify initial factorization of mixture matrix H. Default is None.
       When specified, :param:`seed` must be None.
    :type H: Instance of the :class:`scipy.sparse` sparse matrices types,
       :class:`numpy.ndarray`, :class:`numpy.matrix`, tuple of instances of the
       latter classes or None

    :param rank: The factorization rank to achieve. Default is 30.
    :type rank: `int`

    :param n_run: It specifies the number of runs of the algorithm. Default is
       1. If multiple runs are performed, fitted factorization model with the
       lowest objective function value is retained.
    :type n_run: `int`

    :param callback: Pass a callback function that is called after each run when
       performing multiple runs. This is useful if one wants to save summary
       measures or process the result before it gets discarded. The callback
       function is called with only one argument :class:`models.mf_fit.Mf_fit` that
       contains the fitted model. Default is None.
    :type callback: `function`

    :param callback_init: Pass a callback function that is called after each
       initialization of the matrix factors. In case of multiple runs the function
       is called before each run (more precisely after initialization and before
       the factorization of each run). In case of single run, the passed callback
       function is called after the only initialization of the matrix factors.
       This is useful if one wants to obtain the initialized matrix factors for
       further analysis or additional info about initialized factorization model.
       The callback function is called with only one argument
       :class:`models.mf_fit.Mf_fit` that (among others) contains also initialized
       matrix factors. Default is None.
    :type callback_init: `function`

    :param track_factor: When :param:`track_factor` is specified, the fitted
        factorization model is tracked during multiple runs of the algorithm. This
        option is taken into account only when multiple runs are executed
        (:param:`n_run` > 1). From each run of the factorization all matrix factors
        are retained, which can be very space consuming. If space is the problem
        setting the callback function with :param:`callback` is advised which is
        executed after each run. Tracking is useful for performing some quality or
        performance measures (e.g. cophenetic correlation, consensus matrix,
        dispersion). By default fitted model is not tracked.
    :type track_factor: `bool`

    :param track_error: Tracking the residuals error. Only the residuals from
        each iteration of the factorization are retained. Error tracking is not
        space consuming. By default residuals are not tracked and only the final
        residuals are saved. It can be used for plotting the trajectory of the
        residuals.
    :type track_error: `bool`

    :param update: Type of update equations used in factorization. When specifying
       model parameter ``update`` can be assigned to:

           #. 'euclidean' for classic Euclidean distance update
              equations,
           #. 'divergence' for divergence update equations.
       By default Euclidean update equations are used.
    :type update: `str`

    :param objective: Type of objective function used in factorization. When
       specifying model parameter :param:`objective` can be assigned to:

            #. 'fro' for standard Frobenius distance cost function,
            #. 'div' for divergence of target matrix from NMF
               estimate cost function (KL),
            #. 'conn' for measuring the number of consecutive
               iterations in which the connectivity matrix has not
               changed.
       By default the standard Frobenius distance cost function is used.
    :type objective: `str`

    :param conn_change: Stopping criteria used only if for :param:`objective`
       function connectivity matrix measure is selected. It specifies the minimum
       required of consecutive iterations in which the connectivity matrix has not
       changed. Default value is 30.
    :type conn_change: `int`

    **Stopping criterion**

    Factorization terminates if any of specified criteria is satisfied.

    :param max_iter: Maximum number of factorization iterations. Note that the
       number of iterations depends on the speed of method convergence. Default
       is 30.
    :type max_iter: `int`

    :param min_residuals: Minimal required improvement of the residuals from the
       previous iteration. They are computed between the target matrix and its MF
       estimate using the objective function associated to the MF algorithm.
       Default is None.
    :type min_residuals: `float`

    :param test_conv: It indicates how often convergence test is done. By
       default convergence is tested each iteration.
    :type test_conv: `int`
    """
    def __init__(self, V, seed=None, W=None, H=None, p=None, K=None, rank=30, max_iter=30,
                 min_residuals=1e-5, test_conv=None, n_run=1, callback=None,
                 callback_init=None, track_factor=False, track_error=False,
                 conn_change=30, **options):
        self.name = "cnmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, vars())
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 \
                                              or self.track_error else None

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        for run in range(self.n_run):
            self.W, self.H = self.seed.initialize(
                self.V, self.rank, self.options)
            p_obj = c_obj = sys.float_info.max
            best_obj = c_obj if run == 0 else best_obj
            iter = 0
            if self.callback_init:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback_init(mffit)
      
            #CVX code goes here
            m,n = self.V.shape

            # Uncomment to normalize matrix
            summed = 1./np.sum(self.V, axis=0)
            D= np.diagflat(summed)
            self.V = self.V @ D
            
            x = cvx.Variable([n,n])

            kappa = 0.2
            beta = 2/self.rank
            epsilon = kappa * (1-beta)/((self.rank-1)*(1-beta)+1)

            if self.p is None:
               p = np.random.rand(n,1)
            elif self.p.shape != (n,1) and self.p.shape != (1,n):
               p = np.random.rand(n,1)
            else:
               p = self.p
               if p.shape == (1,n):
                  p = p.T
            
            objective = cvx.Minimize(p.T @ cvx.reshape(cvx.diag(x),(n,1)))
            constraints = [x >= 0]
            for i in range(0, n):
                constraints += [
                cvx.norm((np.reshape(self.V[:,i], (m,1)) - self.V @ cvx.reshape(x[:,i], (n,1))), p=1) <= 2*epsilon,
                x[i,i] <= 1 ]
                for j in range(0,n):
                    constraints += [x[i,j] <= x[i,i]]

            constraints += [cvx.trace(x) == self.rank]
            prob = cvx.Problem(objective, constraints)
            prob.solve()
            print("Problem Value: " + str(prob.value))

            print("X: ")
            print(x.value)
            # Create a copy to make sure it's not immutable
            X = np.array(np.diag(x.value))
            K = []
            for i in range(0, self.rank):
               b = np.unravel_index(np.argmax(X), X.shape)[0]
               K.append(b)
               X[b] = -1

            print("K:")
            print(K)

            self.K = K


            if self.callback:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback(mffit)
            if self.track_factor:
                self.tracker.track_factor(
                    run, W=self.W, H=self.H, final_obj=c_obj, n_iter=iter)
            # if multiple runs are performed, fitted factorization model with
            # the lowest objective function value is retained
            if c_obj <= best_obj or run == 0:
                best_obj = c_obj
                self.n_iter = iter
                self.final_obj = c_obj
                mffit = mf_fit.Mf_fit(copy.deepcopy(self))

        mffit.fit.tracker = self.tracker
        return mffit

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
