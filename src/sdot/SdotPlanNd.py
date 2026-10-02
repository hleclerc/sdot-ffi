"""LA SOLUTION d'un transport semi-discret en dimension `d >= 2` : les poids d'un diagramme de
puissance tels que la masse de chaque cellule contre la densité cible égale la masse du dirac
correspondant.

Elle se demande à un `OtProblem`, et pas autrement :

    sol = OtProblem( SumOfDiracs( pos ), image ).solve()
    sol = ot_solve( SumOfDiracs( pos ), image )        # le même, pour un transport résolu UNE fois

TOUT L'AJUSTEMENT EST EN C++ ( `sdot/sdotplan/` ), en UN `driver.call` : le point de départ, le
Newton amorti, ses diagrammes, le laplacien, le solveur linéaire, l'amortissement. Python ne fait
que poser le problème et lire ce qui en sort. C'est ce que le banc `solvers_des_familles` a
conclu ( README § 3, § 7, § 9, § 10 ) :

  * le NEWTON AMORTI de Kitagawa-Mérigot-Thibert gagne partout, de 2x à 4x en diagrammes comme en
    temps, contre L-BFGS et le gradient conjugué sur le dual, quel que soit leur
    préconditionnement -- la difficulté du transport semi-discret n'est pas la non-linéarité du
    dual, c'est sa NON-RÉGULARITÉ ( les cellules qui se vident ), et un pas de gradient s'y
    écrase comme un pas de Newton, en moins bien ( `sdotplan/Newton.h` ) ;
  * la hessienne est le laplacien du graphe de Laguerre, assemblé SANS TRI depuis les facettes
    du diagramme qui a mesuré le résidu -- un balayage livre les deux ( `sdotplan/Balayage.h`,
    `sdotplan/Laplacien.h` ) ; Cholesky creux en 2D, multigrille algébrique au-delà et en grand
    ( `sdotplan/Lineaire.h` ) ;
  * le pas d'essai repart du dernier pas accepté, jamais de 1 ( dix diagrammes par pas gagnés
    dans la phase linéaire ) ; en 2D, les LIMITES des cellules qui s'écrasent le long de la
    direction remplacent les essais à l'aveugle ( `sdotplan/Limites.h` ).

= Le point de départ

Newton demande un départ ADMISSIBLE ( aucune cellule vide ). Le Voronoï l'est dès que les diracs
sont dans le domaine ; sinon, ou si les poids donnés vident une cellule, le C++ choisit lui-même le
meilleur des trois -- les poids donnés, le Voronoï, la SIMILITUDE qui ramène le nuage dans le
domaine -- et le dit ( `stats[ "depart" ]` ). Voir `sdotplan/Solve.h`.

= Ce que la solution PORTE, et pourquoi ce n'est pas qu'un vecteur de poids

`weights` est la réponse quand les germes sont distincts. Quand deux germes sont CONFONDUS à `1e-8`,
il n'existe pas de couple de `double` qui code le plan qui les sépare à `1e-9` près : le banc l'a
mesuré et calculé ( README § 23.11 ), et c'est pourquoi un solveur dont l'interface est `w` plafonne
vers `1e-6` sur un nuage dégénéré. La solution porte donc AUSSI l'agrégat -- `clusters`,
`cluster_nu` -- même quand il n'y a aucune grappe, pour que la règle soit lisible une fois pour
toutes : si le consommateur veut des CELLULES, il prend le diagramme réduit plus les plans, et tout
est exact ; s'il veut des POIDS, il accepte le plancher `eps |w| / ( 2 delta h )`.

Ce que ça ne coûte pas : le COÛT DE TRANSPORT est aveugle à cette dégénérescence ( § 23.12, écart
relatif `1.2e-17` ), donc tout ce qui est une somme pondérée par les masses -- le coût, la masse
totale, un moment global -- n'a rien à en savoir.

= Un seul diagramme

Les positions sont les CONSTANTES de l'ajustement : le diagramme est bâti UNE FOIS ( son arbre
avec ), et le solveur ne fait que lui POSER les poids essayés -- ce qui, pour un stockage BSP,
refait le majorant des poids de chaque nœud et rien d'autre. Ce qu'il écrit ( les poids triés, les
majorants ) sont des SORTIES de l'appel, reprises par le diagramme après coup : les entrées d'un
appel sont en lecture seule.

`PowerDiagram` et `Cell` sont des OUTILS À LA DEMANDE : `sol.power_diagram()` les fabrique pour
dessiner ou pour une fonctionnelle qu'on n'avait pas prévue. Rien dans la résolution ne passe par
eux côté Python.
"""

import warnings

import loom
from loom.compilation.FfiCode import FfiCode
from loom.drivers.driver import driver
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, Tensor
from loom.util import Aggregate

from .CellScratch import fp_size
from .PowerDiagram import PowerDiagram


# ce que `stats` porte, dans l'ordre de `sdotplan/Solve.h::Stat`
_STATS = [ "fin", "reste", "reste0", "nb_iter", "nb_diag", "nb_recul", "t_maj", "t_diag", "t_asm", "t_lin", "t_lim", "eps",
           "masse_domaine", "nb_deborde", "nb_cell_lim", "nb_tours_essai", "lin_nb_hier", "lin_nb_iter", "lin_pire", "depart", "t_total",
           "nb_etapes", "min_masse_depart" ]
# une ligne de `history`, dans l'ordre de `sdotplan/Solve.h::Hist`
_HISTORY = [ "step", "t", "residual_l2", "min_measure", "max_abs_residual", "nb_diag", "nb_evals", "s" ]
_FIN = { 0: "en cours", 1: "converge", 2: "max iterations", 3: "stagnation", 4: "solveur lineaire en echec" }
_DEPART = { 0: "weights0", 1: "voronoi", 2: "similitude" }
_LIN = { "auto": 0, "cholesky": 1, "amg": 2, "cg": 3 }
_PAS = { "trials": 0, "limits": 1 }
_CONTINUATION = { "never": 0, "auto": 1, "always": 2 }

#: les anciens arguments de `SdotPlanNd( src, dst, ... )`, et où ils vivent maintenant -- lu par le
#: chemin déprécié ( voir `__init__` )
_DEPRECATED_TO_TUNING = ( "accelerator", "memory", "step", "linear_solver", "mass_rtol", "t_min",
                          "max_backtracks", "restart_factor", "conv_start", "conv_ratio", "conv_min",
                          "conv_threshold" )


class _Options( Aggregate ):
    """les réglages du solveur, tels que `sdotplan/Solve.h` les lit"""
    mass_tol       : RealTensor
    mass_rtol      : RealTensor
    t_min          : RealTensor
    mult_ok        : RealTensor
    facteur        : RealTensor
    beta0          : RealTensor
    mult_lim       : RealTensor
    confiance      : RealTensor
    conv_s0        : RealTensor
    conv_ratio     : RealTensor
    conv_min       : RealTensor
    conv_seuil     : RealTensor
    max_iter       : IntTensor
    max_reculs     : IntTensor
    lin            : IntTensor
    pas            : IntTensor
    trace          : IntTensor
    continuation   : IntTensor
    cap0           : IntTensor
    kernel_fp_size : CtShapeVar


class _History( Aggregate ):
    """un pas ACCEPTÉ par ligne ( `step = 0` : le départ ), et les poids de chaque pas si on les a
    demandés ( `weights`, sinon `Unbound` -- un tableau `[ pas, n ]` qu'on ne veut pas toujours )"""
    rows      : RealTensor[ "num_step", "num_hist" ]
    weights   : RealTensor[ "num_step", "num_point" ]
    num_step  : Axis[ "nb_steps" ]
    num_hist  : Axis[ "nb_hist" ]
    num_point : Axis[ "nb_points" ]
    nb_steps  : ShapeVar
    nb_hist   : ShapeVar
    nb_points : ShapeVar


class SdotPlanNd:
    """voir la docstring du module"""

    # -- comment on en obtient une -------------------------------------------------------------

    @classmethod
    def _solve( cls, problem, settings, verbose, warm = None ):
        """LE chemin : `OtProblem.solve()` et lui seul passe par ici. `warm` sont les poids que le
        problème a gardés de sa dernière solution -- le départ quand `settings.weights0` est `None`."""
        self = cls.__new__( cls )
        self._build( problem, settings, verbose, warm )
        return self

    def __init__( self, src_dist, dst_dist, verbose = False, **kwargs ):
        """DÉPRÉCIÉ -- `OtProblem( src_dist, dst_dist ).solve( Iterative( ... ) )`.

        Il y a maintenant une seule porte d'entrée : un problème se pose ( `OtProblem` ), puis on
        lui demande une solution. Ce constructeur traduit les anciens arguments et sera retiré."""
        warnings.warn( "SdotPlanNd( src, dst, ... ) est deprecie : poser le probleme puis le "
                       "resoudre -- OtProblem( src, dst ).solve( Iterative( ... ) ). Voir la "
                       "docstring d'`OtProblem`.", DeprecationWarning, stacklevel = 2 )
        from .OtProblem import Iterative, OtProblem, Tuning

        kw = dict( kwargs )
        if "kernel_dtype" in kw:
            kd = kw.pop( "kernel_dtype" )
            kw[ "precision" ] = { None: "auto", "FP64": "fp64", "FP32": "fp32" }.get( kd, kd )
        if "mass_tol" in kw:
            kw[ "tol" ] = kw.pop( "mass_tol" )
        tuning = Tuning( **{ k: kw.pop( k ) for k in _DEPRECATED_TO_TUNING if k in kw } )
        self._build( OtProblem( src_dist, dst_dist ), Iterative( tuning = tuning, **kw ), verbose )

    # -- ce que l'appel fait -------------------------------------------------------------------

    def _build( self, problem, settings, verbose, warm = None ):
        # le solveur est du code HOTE sur la file CPU ( `sdotplan/Balayage.h` ) : un driver dont le device
        # est un GPU ne peut pas l'appeler aujourd'hui -- `LOOM_DEVICE=cpu`, ou un driver CPU
        if not driver.device.is_cpu:
            raise NotImplementedError( "SdotPlanNd : le solveur tourne sur le CPU pour l'instant ( les balayages et le "
                                       "solveur lineaire sont du code hote ) ; choisir le device CPU ( LOOM_DEVICE=cpu )" )
        tun = settings.tuning
        #: le problème dont on est la solution
        self.problem = problem
        src_dist, dst_dist = problem.source, problem.target
        d = problem.nb_dims

        # le domaine : le support de la densité, qui doit le borner ( lève sinon ). `PowerDiagram`
        # l'ajoute lui-même depuis la distribution -- on le DEMANDE ici pour que le refus soit dit
        # avant qu'on ait alloué quoi que ce soit.
        problem.domain

        # le pas par les limites n'existe qu'en 2D ( `sdotplan/Limites.h` ) ; ailleurs, les essais
        step = { "auto": "limits" if d == 2 else "trials" }.get( tun.step, tun.step )
        if step == "limits" and d != 2:
            raise ValueError( "step = 'limits' : 2D seulement pour l'instant ( voir `sdotplan/Limites.h` )" )
        if step not in _PAS:
            raise ValueError( f"step inconnu : { tun.step !r } ( 'auto', 'trials' ou 'limits' )" )
        if tun.linear_solver not in _LIN:
            raise ValueError( f"linear_solver inconnu : { tun.linear_solver !r } ( { ', '.join( _LIN ) } )" )

        # LE diagramme, bâti une fois sur les positions ( voir la docstring du module ) ; les poids
        # qu'il porte à un instant donné sont les derniers posés
        w0_given = settings.weights0 if settings.weights0 is not None else warm
        self._pd = PowerDiagram( src_dist.positions,
                                 RealTensor[ src_dist.num_dirac ].full( 0.0 ) if w0_given is None else w0_given,
                                 accelerator = tun.accelerator, kernel_dtype = settings.kernel_dtype,
                                 distribution = dst_dist, memory = tun.memory,
                                 scratch_capacity = tun.scratch_capacity )
        pd = self._pd
        n = int( pd.nb_points.value )

        #: les masses cibles, indexées comme les cellules
        self._masses = RealTensor[ pd.num_point ]( src_dist.weights.raw )

        options = _Options(
            mass_tol = float( settings.tol ), mass_rtol = float( tun.mass_rtol ), t_min = float( tun.t_min ),
            mult_ok = float( tun.restart_factor ), facteur = 0.9, beta0 = 0.25, mult_lim = 2.0, confiance = 0.0,
            conv_s0 = float( tun.conv_start or 0.0 ), conv_ratio = float( tun.conv_ratio ),
            conv_min = float( tun.conv_min or 0.0 ), conv_seuil = float( tun.conv_threshold ),
            max_iter = int( settings.max_iter ), max_reculs = int( tun.max_backtracks ), lin = _LIN[ tun.linear_solver ],
            pas = _PAS[ step ], trace = int( bool( verbose ) ), continuation = _CONTINUATION[ settings.continuation ],
            cap0 = int( pd._scratch_capacity ),
            kernel_fp_size = fp_size( pd.kernel_dtype ),
        )

        weights = RealTensor[ pd.num_point ]()
        history = _History( nb_hist = len( _HISTORY ), nb_points = n )
        # LES AXES SONT CEUX DU DIAGRAMME, et c'est la seule chose a ne pas rater ici : un axe
        # `num_point` fabrique a part porte le meme NOM sans etre le meme, et
        # `cell_masses * ( positions - barycenters )` devient alors un produit exterieur `[ n, n, d, d ]`
        # au lieu du gradient. Ils se declarent donc depuis `pd`.
        cell_masses = RealTensor[ pd.num_point ]()
        barycenters = RealTensor[ pd.num_point, pd.dim ]()
        cost        = RealTensor()
        stats = RealTensor[ Axis( ShapeVar( len( _STATS ) ), name = "num_stat" ) ]()
        w0 = RealTensor[ pd.num_point ]( pd.weights.raw )

        dom = pd._domain_cell()
        dist_expr, _, dist_kwargs = pd._dist_for()
        pd_expr, pd_kwargs, pd_produced = pd._solver_weights_call()

        loom.ffi_call(
            "sdotplan_solve",
            # `handler` et pas le noyau echafaude par defaut : ce corps EST le handler. Il est du
            # code HOTE -- il a besoin de la `queue`, et il pilote lui-meme son parallelisme ( cent
            # diagrammes dans un seul appel ), donc il n'y a ni foncteur par item ni `run_parallel`
            # a engendrer autour de lui. Voir la docstring de `FfiCode`.
            # `inline` : le corps EST celui du handler, loom n'ecrit que son enveloppe
            # ( `void kernel( queue, batch_axes, args )` ). Il est du code HOTE -- il a besoin de la
            # `queue`, et il pilote lui-meme son parallelisme ( cent diagrammes dans un seul appel ),
            # donc il n'y a ni foncteur par item ni `run_parallel` a engendrer autour de lui.
            FfiCode.inline( includes = [ "sdot/sdotplan/Solve.h" ],
                sources = [ "sdot/sdotplan/Lineaire.cpp" ],
                code = "\n".join( [
                    # les expressions que `PowerDiagram` engendre nomment `inputs` / `outputs` ( la forme
                    # d'un noyau par item ) : on les RETROUVE ici sous leur nom, et rien n'a a le savoir
                    "auto &inputs = args.inputs; auto &outputs = args.outputs;",
                    "using TK_sdotplan = std::conditional_t<CT_VALUE( inputs.options.kernel_fp_size ) == 64, double, float>;",
                    f"auto pd_sdotplan = { pd_expr };",
                    "sdotplan::OptionsSolveur os;",
                    "sdotplan::NewtonOptions &no = os.newton;",
                    "no.tol_abs = double( inputs.options.mass_tol ); no.tol_rel = double( inputs.options.mass_rtol ); no.t_min = double( inputs.options.t_min );",
                    "no.mult_ok = double( inputs.options.mult_ok ); no.facteur = double( inputs.options.facteur ); no.beta0 = double( inputs.options.beta0 );",
                    "no.mult_lim = double( inputs.options.mult_lim ); no.confiance = double( inputs.options.confiance );",
                    "no.maxit = int( SI( inputs.options.max_iter ) ); no.max_reculs = int( SI( inputs.options.max_reculs ) );",
                    "no.pas = int( SI( inputs.options.pas ) ); no.trace = SI( inputs.options.trace ) != 0;",
                    "os.lin = sdotplan::Lin( int( SI( inputs.options.lin ) ) ); os.cap0 = SI( inputs.options.cap0 );",
                    "os.continuation = int( SI( inputs.options.continuation ) ); os.seuil_continuation = double( inputs.options.conv_seuil );",
                    "os.conv_s0 = double( inputs.options.conv_s0 ); os.conv_ratio = double( inputs.options.conv_ratio ); os.conv_min = double( inputs.options.conv_min );",
                    f"sdotplan::resoudre<TK_sdotplan>( queue, pd_sdotplan, inputs.power_diagram, inputs.dom_cell, { dist_expr }, inputs.nu, inputs.w0, os, "
                    "outputs.weights, outputs.history, outputs.stats, outputs.cell_masses, outputs.barycenters, outputs.cost );",
                ] ) ),
            power_diagram = pd,
            dom_cell = dom,
            nu = self._masses,
            w0 = w0,
            options = options,
            weights = loom.out( weights ),
            # `nb_steps` est ECRIT par le noyau ( le nombre de pas acceptes ), donc il doit etre
            # nomme : un `ShapeVar` qu'on ne declare pas reste non lie, et le C++ ne voit qu'une vue
            # nulle. Les poids de chaque pas, eux, ne sont ecrits que si on les a demandes --
            # `weights` non nomme reste observe, donc ni alloue ni lu.
            history = loom.out( history, writes = ( [ "rows", "nb_steps", "weights" ] if settings.keep_weights
                                                    else [ "rows", "nb_steps" ] ),
                                capacities = { "nb_steps": int( settings.max_iter ) + 1 } ),
            stats = loom.out( stats ),
            cell_masses = loom.out( cell_masses ),
            barycenters = loom.out( barycenters ),
            cost = loom.out( cost ),
            has_dynamic_capacity = False,
            **pd_kwargs,
            **dist_kwargs,
        )
        pd._solver_weights_after( pd_produced )

        #: les poids AJUSTÉS, `[ n ]`, indexés comme `positions` ( `weights[ 0 ] == 0` : la jauge )
        self.weights = weights
        #: la mesure de chaque cellule aux poids AJUSTÉS, `[ n ]` -- celle que Newton a mesurée
        self.cell_masses = cell_masses
        #: le barycentre de chaque cellule, `[ n, d ]` ( son germe si elle est vide )
        self.barycenters = barycenters
        #: le coût de transport `W_2^2` ( un `Tensor` scalaire : `float( sol.cost )` pour le nombre )
        self.cost = cost
        self._read_stats( stats, settings )
        self._read_history( history, settings, pd )

    def _read_stats( self, stats, settings ):
        #: ce que le solveur rapporte ( voir `sdotplan/Solve.h::Stat` ), plus `fin` et `depart` en clair
        st = stats.raw
        self.stats = { name: float( st[ k ] ) for k, name in enumerate( _STATS ) }
        self.stats[ "fin" ] = _FIN.get( int( self.stats[ "fin" ] ), "?" )
        self.stats[ "depart" ] = _DEPART.get( int( self.stats[ "depart" ] ), "?" )
        for name in ( "nb_iter", "nb_diag", "nb_recul", "nb_deborde", "nb_cell_lim", "nb_tours_essai",
                      "lin_nb_hier", "lin_nb_iter", "nb_etapes" ):
            self.stats[ name ] = int( self.stats[ name ] )
        # l'agrégation : DEMANDÉE ici, pas encore faite par le C++ ( étape 7 de
        # `notes/2026-10-02-sdotplan.md` ) -- et c'est dit plutôt que tu, parce qu'un nuage dégénéré
        # plafonne alors vers `1e-6` sans le moindre message ( README § 23.11 ).
        self.stats[ "agregation" ] = ( "demandee, pas encore branchee ( etape 7 )" if settings.aggregate
                                       else "non demandee" )

    def _read_history( self, history, settings, pd ):
        #: un dict par pas ACCEPTÉ -- `step = 0` est le point de départ : `t`, `residual_l2`,
        #: `min_measure`, `max_abs_residual`, `nb_diag` ( cumulé ), `nb_evals` ( les diagrammes de
        #: ce pas ), et `weights` si `keep_weights`
        nb_steps = int( history.nb_steps.value )
        rows = history.rows.raw[ :nb_steps ]
        self.history = []
        for s in range( nb_steps ):
            entry = { name: float( rows[ s, k ] ) for k, name in enumerate( _HISTORY ) }
            entry[ "step" ] = int( entry[ "step" ] )
            if settings.keep_weights:
                entry[ "weights" ] = RealTensor[ pd.num_point ]( history.weights.raw[ s ] )
            self.history.append( entry )

    # -- ce que la solution dit ----------------------------------------------------------------

    @property
    def converged( self ):
        return self.stats[ "fin" ] == "converge"

    @property
    def clusters( self ):
        """À quelle GRAPPE chaque dirac appartient, ou `None` quand aucun n'a été fusionné.

        `None` est le cas courant et il veut dire « les poids suffisent ». Dès qu'il y a des
        grappes, c'est le couple ( diagramme réduit, plans de coupe ) qui porte la précision et non
        `weights` -- voir la docstring du module et le README § 23.11."""
        return None

    @property
    def target_masses( self ) -> Tensor:
        """`nu` : la masse cible de chaque cellule ( les masses des diracs, normalisées, puis remises
        à l'échelle de ce que le domaine contient -- voir `sdotplan/Solve.h` )"""
        return self._masses * ( self.stats[ "masse_domaine" ] / float( self._masses.sum() ) )

    def residual( self, weights = None ) -> Tensor:
        """`mesure_i( weights ) - nu_i` -- ZÉRO au point cherché, DÉRIVABLE par rapport à `weights`
        ( voir `PowerDiagram.measures` ). Un diagramme de plus : c'est un outil de diagnostic, pas
        une sortie de la résolution ( `cell_masses` l'est )."""
        w = self.weights if weights is None else weights
        return self.power_diagram( w ).measures - self.target_masses

    def power_diagram( self, weights = None ) -> PowerDiagram:
        """L'OUTIL À LA DEMANDE : le `PowerDiagram` pour `weights` ( par défaut : les poids AJUSTÉS )
        -- toujours le MÊME objet, auquel on pose ces poids-là. Pour dessiner, pour inspecter, pour
        une fonctionnelle qu'on n'avait pas prévue ; rien dans la résolution ne passe par lui."""
        self._pd.weights = self.weights if weights is None else weights
        return self._pd

    # -- ce que le plan VAUT -------------------------------------------------------------------

    def transport( self ):
        """`( cost, barycenters, masses )` aux poids AJUSTÉS : le coût `W_2^2 = sum_i int_{cell_i}
        |x - p_i|^2 rho` ( un `Tensor` scalaire ), le barycentre de chaque cellule ( `[ n, d ]` ) et
        sa masse ( `[ n ]` ).

        Les trois sont des SORTIES de l'appel -- le solveur les mesure sur les poids ajustés, dans le
        même C++ et avec le même scratch. Une cellule vide garde son germe pour barycentre."""
        return self.cost, self.barycenters, self.cell_masses

    def cost_and_position_grad( self ):
        """`( cost, grad )` : le coût, et sa dérivée par rapport aux POSITIONS des diracs
        ( `[ n, d ]` ) -- par le théorème de l'enveloppe : aux poids optimaux, la dérivée du coût
        par rapport à `p_i` ne passe pas par les cellules, et vaut `2 m_i ( p_i - b_i )`, `b_i` le
        barycentre de la cellule et `m_i` sa masse ( qui est la masse cible du dirac ). C'est la
        même formule que `SdotPlan1d`, et ce qu'une reconstruction consomme ( `otrec` )."""
        return self.cost, 2 * self.cell_masses * ( self._pd.positions - self.barycenters )
