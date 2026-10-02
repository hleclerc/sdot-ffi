"""`OtProblem` : CE QU'ON RÉSOUT -- et la seule porte d'entrée des plans de transport semi-discrets.

= Deux objets, et pas un de plus

`OtProblem` porte les ENTRÉES, et rien d'autre : deux distributions. Aucun réglage de solveur,
aucun tenseur alloué, aucun arbre, aucune capacité de scratch. Le construire ne déclenche aucun
appel. On peut rebrancher une entrée et redemander la solution :

    pb = OtProblem( SumOfDiracs( pos ), image )
    sol = pb.solve()
    pb.source = SumOfDiracs( pos_suivantes )      # une entrée change
    sol = pb.solve()                              # et la SOLUTION d'avant sert de départ

`solve()` rend une SOLUTION, un objet à part -- `SdotPlan1d` ou `SdotPlanNd`. Un problème peut
avoir plusieurs solutions ( deux tolérances, deux départs ) et ne doit donc pas en porter une.

Pour un transport qu'on ne résout QU'UNE FOIS, `ot_solve( source, target, ... )` fait les deux en
une expression. Ce n'est un raccourci que dans ce cas-là : dès qu'on résout plusieurs fois, c'est
l'`OtProblem` gardé qui porte la SOLUTION de la fois d'avant, et c'est lui qui fait qu'un transport
voisin coûte quelques pas de Newton au lieu de quelques dizaines. Un plan et non des poids, parce
que des poids seuls ne décrivent pas une solution dès qu'il y a des germes confondus ( § 23.11 ).

= Direct ou itératif : le problème a déjà choisi

    d = 1, diracs contre densité  ->  TRI plus inversion de fonction de répartition.  `SdotPlan1d`
    d >= 2                        ->  POINT FIXE d'un Newton amorti.                  `SdotPlanNd`

Le premier est exact, sans boucle, sans tolérance et sans point de départ ; il est dérivable de
bout en bout et batché sur des milliers d'angles. Le second est une boucle, avec une tolérance, un
départ, un historique, et un gradient par le théorème de l'enveloppe.

**Les paramètres ne diffèrent donc pas parce que l'appelant choisit une méthode : ils diffèrent
parce que le RÉGIME diffère, et le régime se lit sur les entrées.** D'où la règle : il y a un objet
de réglages PAR RÉGIME, `Direct` et `Iterative`, et **son type EST le régime**. En donner un qui ne
correspond pas est une erreur, pas un argument ignoré en silence :

    sol = pb.solve()                              # les défauts du régime, qui sont les bons
    sol = pb.solve( Iterative( tol = 1e-10 ) )    # d >= 2
    sol = pb.solve( Direct( with_barycenters = True ) )   # d = 1

= Les réglages du banc ne sont pas des réglages d'utilisateur

Tout ce que `solvers_des_familles` fait varier pour COMPARER des algorithmes -- le choix du pas, le
solveur linéaire, l'échelle de continuation, l'accélérateur spatial, la mémoire des voisins -- a un
défaut MESURÉ, et un utilisateur qui le change choisit mal. Ces réglages existent, derrière une
porte de service : `Iterative( tuning = Tuning( ... ) )`. `Tuning` est explicitement INSTABLE --
c'est la surface par laquelle le banc continue d'exister, et rien d'autre ne doit en dépendre.

Et la leçon à ne pas réapprendre ( README § 24.5 ) : **un défaut appartient au régime où il a été
mesuré.** `residu = log` gagne sur les solves directs et casse la continuation en densité.
"""

import numpy as np                  # les seuls tableaux hôtes d'ici : les quelques plans du domaine


_PRECISIONS = { "auto": "FP64", "fp64": "FP64", "fp32": "FP32" }


class Tuning:
    """LES RÉGLAGES DU BANC. Instable, non supporté, et rien dans `sdot` ne doit en dépendre.

    Exactement les paramètres retirés de la surface publique ( voir le module ) : ils gardent leur
    défaut mesuré, et la section du banc qui l'a mesuré est écrite à côté.
    """

    def __init__( self,
                  # le pas, et le solveur linéaire -- `auto` = ce que le § 24.4 conclut
                  step = "auto", linear_solver = "auto",
                  # l'échelle de la continuation en largeur ( § 9.2 : le ratio sqrt( 2 ) est mesuré )
                  conv_start = None, conv_ratio = 2 ** 0.5, conv_min = None, conv_threshold = 1e-2,
                  # les garde-fous de l'amortissement ( § 3 : `restart_factor = 4` est mesuré )
                  t_min = 1e-10, max_backtracks = 60, restart_factor = 4.0, mass_rtol = 0.0,
                  # l'agrégation ( § 23.5 : le mal commence vers 0.2 % de l'espacement médian )
                  delta_aggregation = None,
                  # la machine : l'accélérateur spatial, la mémoire des voisins ( § 11 ), le scratch ( § 18.2 )
                  accelerator = None, memory = None, scratch_capacity = None ):
        self.step              = step
        self.linear_solver     = linear_solver
        self.conv_start        = conv_start
        self.conv_ratio        = conv_ratio
        self.conv_min          = conv_min
        self.conv_threshold    = conv_threshold
        self.t_min             = t_min
        self.max_backtracks    = max_backtracks
        self.restart_factor    = restart_factor
        self.mass_rtol         = mass_rtol
        self.delta_aggregation = delta_aggregation
        self.accelerator       = accelerator
        self.memory            = memory
        self.scratch_capacity  = scratch_capacity


class Iterative:
    """Les réglages du RÉGIME ITÉRATIF ( `d >= 2`, `SdotPlanNd` ) -- et son type dit le régime.

    `tol` : on s'arrête dès que `max_i | m_i - nu_i | <= tol`, ABSOLU, dans l'unité des masses
    normalisées ( la masse cible d'un dirac est `1 / n` ). `max_iter` : pas de Newton, au plus.

    `ot_plan` : LE DÉPART, sous la forme d'une SOLUTION précédente -- ce dont vit une reconstruction,
    où un transport voisin coûte quelques pas de Newton au lieu de quelques dizaines. C'est un plan et
    non un vecteur de poids, et c'est structurel : dès qu'il y a des germes confondus, les poids seuls
    NE DÉCRIVENT PAS la solution ( § 23.11 -- c'est le couple ( problème réduit, plans de coupe ) qui
    porte la précision ), donc un départ qui n'est que `w` perd l'agrégat. Un plan le porte, et il
    porte aussi les POSITIONS pour lesquelles il a été résolu : quand elles n'ont pas changé, les
    grappes n'ont pas à être redétectées ( ce que l'étape 7 exploitera ).

    `weights0` : les poids NUS, quand c'est tout ce qu'on a ( un fichier, un essai délibéré ).
    `ot_plan` est la bonne façon ; donner les deux lève. Sans l'un ni l'autre, `OtProblem` propose la
    dernière solution qu'il a rendue ( voir `OtProblem.solve` ). Un départ qui vide une cellule n'est
    pas une erreur : le C++ compare lui-même les départs qu'il connaît et garde le meilleur
    ( `stats[ "depart" ]`, et `stats[ "repris" ]` dit ce que le départ à chaud a fourni ).

    `continuation` : la CONTINUATION EN LARGEUR -- résoudre d'abord pour la densité convolée par une
    gaussienne large, puis de plus en plus étroite, chaque étape partant des poids de la précédente.
    C'est ce qu'il faut à une densité qui se concentre ( des bosses étroites, des déserts où des
    cellules n'ont pas de masse, et où Newton direct STAGNE -- § 9.1 ). `"auto"` la déclenche quand
    le départ laisse une cellule sans masse ; `"always"` / `"never"`.

    `precision` : le flottant dans lequel la géométrie se coupe. `"auto"` est `FP64` : le banc l'a
    mesuré ( § 4 ), l'amortissement demande une décroissance stricte du résidu que le bruit d'une
    aire en `float` refuse bien avant la tolérance. La bascule `fp32 -> fp64` est STRUCTURELLE
    ( § 19.10 ) et appartient donc au C++, pas à l'appelant.

    `aggregate` : FUSIONNER les diracs trop proches avant de résoudre ( § 23.6 ). Allumé, parce que
    l'alternative est un plancher SILENCIEUX vers `1e-6` sur un nuage dégénéré : sans lui,
    `lignes sigma = 0.005` stagne à `2.35e-6` en 113 diagrammes ; avec, il converge à `2.00e-7` en
    78. La détection coûte 4 % d'un diagramme en 2D, 1 % en 3D ( § 23.8 ). La solution porte alors
    l'agrégat, et c'est pourquoi `SdotPlanNd` a des grappes même quand il n'y en a pas ( § 23.11 ).

    `keep_weights` : garder les poids de CHAQUE pas dans `history`, pour rejouer la descente ( un
    tableau `[ pas, n ]`, qu'on ne veut pas toujours ).
    """

    regime = "iterative"

    def __init__( self, tol = 1e-8, max_iter = 100, ot_plan = None, weights0 = None,
                  continuation = "auto", precision = "auto", aggregate = True, keep_weights = False,
                  tuning = None ):
        if precision not in _PRECISIONS:
            raise ValueError( f"precision inconnue : { precision !r } ( { ', '.join( _PRECISIONS ) } )" )
        if continuation not in ( "auto", "always", "never" ):
            raise ValueError( f"continuation inconnue : { continuation !r } ( 'auto', 'always' ou 'never' )" )
        if ot_plan is not None and weights0 is not None:
            raise ValueError( "Iterative : `ot_plan` ET `weights0` -- il n'y a qu'un depart. `ot_plan` "
                              "est celui a garder ( il porte l'agregat et les positions, voir la docstring )" )
        self.ot_plan      = ot_plan
        self.tol          = float( tol )
        self.max_iter     = int( max_iter )
        self.weights0     = weights0
        self.continuation = continuation
        self.precision    = precision
        self.aggregate    = bool( aggregate )
        self.keep_weights = bool( keep_weights )
        self.tuning       = tuning or Tuning()

    @property
    def kernel_dtype( self ):
        return _PRECISIONS[ self.precision ]


class Direct:
    """Les réglages du RÉGIME DIRECT ( `d = 1`, `SdotPlan1d` ) -- et son type dit le régime.

    Il n'y a ni tolérance ni nombre d'itérations : la solution est un tri plus une inversion de
    fonction de répartition, donc exacte. Reste à dire ce qu'on veut en SORTIR.

    `with_barycenters` : produire ET stocker les barycentres ( `[ n, d ]` par élément de batch,
    `80 Go` à l'échelle d'une reconstruction ). Éteint par défaut : l'adjoint sait les recalculer.
    """

    regime = "direct"

    def __init__( self, with_barycenters = False ):
        self.with_barycenters = bool( with_barycenters )


class OtProblem:
    """voir la docstring du module"""

    def __init__( self, source, target ):
        """`source` : la distribution DISCRÈTE -- une `SumOfDiracs` ( ou une `ProjectedSumOfDiracs` ) ;
        ses `weights`, normalisés, sont les masses cibles des cellules.

        `target` : la distribution CONTINUE contre laquelle intégrer ( `Image`, `SumOfGaussians`, ... ).

        LE DOMAINE VIENT DE `target`, ET D'ELLE SEULE : le support qu'elle déclare
        ( `bounding_half_spaces` -- le pavé d'une image, `centres +- 6 sigma` pour des gaussiennes ),
        qui doit être BORNÉ. Les diracs n'y sont pour rien : leurs cellules peuvent être loin d'eux.
        Leur enveloppe ne sert qu'au DÉPART ( le recadrage ). C'est pourquoi `domain` est en LECTURE
        SEULE -- il n'y a pas de second endroit où le dire."""
        if target is None:
            raise ValueError( "OtProblem : il faut une cible -- c'est elle qui donne le domaine" )
        self._last = None                                # la dernière solution rendue ( pas ses poids )
        self._source = None
        self._target = None
        self.source = source
        self.target = target

    # -- les entrées, paramétrables -----------------------------------------------------------

    @property
    def source( self ):
        """la distribution discrète, normalisée"""
        return self._source

    @source.setter
    def source( self, source ):
        if source is None:
            raise ValueError( "OtProblem : il faut une source" )
        source = source.normalized_version()
        # un nuage qui a changé de TAILLE périme la solution gardée ( un étage de multi-échelle )
        if self._source is not None and self._nb_diracs_of( source ) != self._nb_diracs_of( self._source ):
            self._last = None
        self._source = source

    @property
    def target( self ):
        """la densité cible, normalisée -- et ce qui donne le domaine"""
        return self._target

    @target.setter
    def target( self, target ):
        if target is None:
            raise ValueError( "OtProblem : il faut une cible -- c'est elle qui donne le domaine" )
        self._target = target.normalized_version()

    # -- ce qui se LIT sur les entrées --------------------------------------------------------

    @property
    def nb_dims( self ):
        return int( self._source.nb_dims.value )

    @property
    def nb_diracs( self ):
        return self._nb_diracs_of( self._source )

    @property
    def regime( self ):
        """`"direct"` ( un tri, `d = 1` ) ou `"iterative"` ( un Newton amorti, `d >= 2` ) -- DÉDUIT,
        jamais choisi. Voir la docstring du module."""
        return "direct" if self.nb_dims == 1 else "iterative"

    @property
    def domain( self ):
        """Le domaine, en demi-espaces `( directions, offsets )` : `direction . x <= offset`. Il vient
        du support que la CIBLE déclare, et d'elle seule. Lève si ce support ne borne pas."""
        d = self.nb_dims
        support = self._target.bounding_half_spaces()
        if support is not None:
            dirs = np.asarray( support[ 0 ], dtype = float ).reshape( -1, d )
            offs = np.asarray( support[ 1 ], dtype = float ).reshape( -1 )
            if self._bounds( dirs, offs, d ):
                return dirs, offs
        raise ValueError( "OtProblem : le support de la cible ne borne pas le domaine "
                          "( `bounding_half_spaces` ) -- c'est a la cible de le declarer "
                          "( `SumOfGaussians( support_sigmas = ... )` )" )

    # -- resoudre ------------------------------------------------------------------------------

    def solve( self, settings = None, verbose = False ):
        """La SOLUTION : un `SdotPlan1d` si le régime est direct, un `SdotPlanNd` s'il est itératif.

        `settings` : `None` pour les défauts du régime, ou un `Direct` / `Iterative` -- et son TYPE
        doit être celui du régime, sans quoi on lève ( voir la docstring du module ).

        Un `Iterative` qui n'impose pas de départ ( ni `ot_plan` ni `weights0` ) repart de la
        dernière SOLUTION de CE problème, quand il y en a une et que le nombre de diracs n'a pas
        changé : c'est ce dont vit une reconstruction, et ça n'a plus à être recopié à la main par
        l'appelant."""
        regime = self.regime
        if settings is None:
            settings = { "direct": Direct, "iterative": Iterative }[ regime ]()
        elif getattr( settings, "regime", None ) != regime:
            want = { "direct": "Direct", "iterative": "Iterative" }[ regime ]
            raise TypeError( f"OtProblem.solve : ce probleme se resout en regime { regime !r } "
                             f"( nb_dims = { self.nb_dims } ), donc ses reglages sont un `{ want }` "
                             f"et non un `{ type( settings ).__name__ }` -- voir la docstring de `OtProblem`" )

        if regime == "direct":
            from .SdotPlan1d import SdotPlan1d
            return SdotPlan1d._solve( self, settings, verbose )

        from .SdotPlanNd import SdotPlanNd
        # le RE-ECHAUFFEMENT : la derniere SOLUTION, quand l'appelant n'impose pas de depart.
        # Un plan et non des poids -- il porte l'agregat, que `w` seul ne sait pas decrire
        # ( README § 23.11 ). Passe a part, et non ecrit dans `settings` : un objet de reglages que
        # l'appelant garde ne doit pas se mettre a porter l'etat du probleme.
        sol = SdotPlanNd._solve( self, settings, verbose, warm = self._last )
        self._last = sol
        return sol

    # -- les détails -----------------------------------------------------------------------------

    @staticmethod
    def _nb_diracs_of( dist ):
        if not getattr( dist, "_is_dirac_source", False ):
            raise TypeError( f"OtProblem : { type( dist ).__name__ } n'est pas une source discrete "
                             "-- c'est la source qui porte les masses cibles des cellules" )
        return int( dist.nb_diracs.value )

    def _bounds( self, dirs, offs, d ):
        """Est-ce que ces demi-espaces bornent ? Un pavé se reconnaît sans rien construire ; sinon on
        monte le polytope pour le savoir ( quelques plans, côté hôte )."""
        from .PowerDiagram import axis_aligned_box
        if axis_aligned_box( dirs, offs ) is not None:
            return True
        from .Cell import Cell
        dom = Cell.make_unbounded( d, kernel_dtype = "FP64" )
        for k in range( len( offs ) ):
            dom.cut( dirs[ k ], float( offs[ k ] ) )
        return bool( dom.is_bounded )


# -- LE RACCOURCI ------------------------------------------------------------------------------

def ot_solve( source, target, *, verbose = False, **settings ):
    """Poser le problème et le résoudre, en une expression -- pour un transport qu'on ne résout
    QU'UNE FOIS.

        sol = ot_solve( SumOfDiracs( pos ), image )
        sol = ot_solve( SumOfDiracs( pos ), image, tol = 1e-10 )     # d >= 2 : un `Iterative`
        sol = ot_solve( SumOfDiracs( pos2 ), image, ot_plan = sol )   # en repartant du plan d'avant
        sol = ot_solve( diracs_1d, image_1d, with_barycenters = True )   # d = 1 : un `Direct`

    Les réglages se donnent ici en MOTS-CLEFS, et non comme un objet : le régime est déduit des
    entrées (voir la docstring du module), donc la classe de réglages qui les reçoit l'est aussi.
    Un nom qui n'appartient pas au régime lève, en disant lequel c'est et ce qu'il accepte --
    jamais ignoré en silence.

    Pour repartir d'une résolution précédente, `ot_plan = la_solution_d_avant` -- un PLAN et non des
    poids, parce que les poids seuls ne décrivent pas une solution dès qu'il y a des germes confondus
    ( voir `Iterative` ).

    **Mais dans une boucle, garder le problème est mieux.** Il propose la dernière solution tout seul,
    il la périme tout seul quand le nuage change de taille, et c'est un `ot_plan` de moins à faire
    circuler. Une reconstruction, une descente, un balayage de paramètre gardent donc le problème et
    rebranchent ses entrées :

        pb = OtProblem( SumOfDiracs( pos ), image )
        for _ in range( nb_pas ):
            sol = pb.solve()                      # repart des poids d'avant
            pb.source = SumOfDiracs( deplace( pos, sol ) )

    `ot_solve` refait un `OtProblem` neuf à chaque appel : sans `ot_plan`, il repart de zéro."""
    pb = OtProblem( source, target )
    cls = { "direct": Direct, "iterative": Iterative }[ pb.regime ]
    try:
        reglages = cls( **settings )
    except TypeError as e:
        import inspect
        noms = [ n for n in inspect.signature( cls ).parameters if n != "self" ]
        raise TypeError( f"ot_solve : ce probleme se resout en regime { pb.regime !r } "
                         f"( nb_dims = { pb.nb_dims } ), donc ses reglages sont ceux de `{ cls.__name__ }` "
                         f"( { ', '.join( noms ) } ) -- { e }" ) from None
    return pb.solve( reglages, verbose )
