from netgen.geom2d import unit_square
from netgen.occ import WorkPlane, Rectangle, Circle, MoveTo, Glue, X, Y, Z, OCCGeometry, Axis
import ngsolve as ngs
import numpy as np


class NGSGeo:
    def __init__(self, maxh, **kwargs):
        self.maxh = maxh
        self.outer_bnd = "outer"

    def make(self):
        result = self.geo.GenerateMesh(maxh=self.maxh)
        if isinstance(result, ngs.Mesh):
            ngs_mesh = result
        else:
            ngs_mesh = ngs.Mesh(result)
        ngs_mesh.Curve(3)
        self.mesh = Mesh(ngs_mesh)


class NGSSimple(NGSGeo):
    """Plain unit square (no internal subdomain). BC names: bottom|right|top|left."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mesh = Mesh(unit_square.GenerateMesh(maxh=self.maxh))

        self.corners = ((0,0), (0,1), (1,1), (1,0))
        self.centre = (0.5, 0.5)

        self.outer_bnd = "left|right|bottom|top"


class NGSCircle(NGSGeo):
    """Outer circle with inner circular subdomain and rectangular scatterer.
    BC names: outer (exterior boundary), interface (inner circle), scat (rectangle).
    """
    def __init__(self, size=0.3, **kwargs):
        super().__init__(**kwargs)
        size = size
        size = 0.3
        centre = 0.5

        # Outer disk
        outer = WorkPlane().Circle(centre, centre, 1.0).Face()
        outer.edges.name = "outer"

        # Inner disk
        inner = WorkPlane().Circle(centre, centre, size).Face()
        inner.edges.name = "interface"

        # Rectangle scatterer (punched out of annulus)
        scat = MoveTo(centre + 1.2 * size, 0.25).Rectangle(0.05, 0.6).Face()
        scat.edges.name = "scat"

        # Annulus = outer disk − inner disk − scatterer
        annulus = (outer - inner) - scat
        annulus.faces.name = "outer"

        inner.faces.name = "inner"

        shape = Glue([annulus, inner])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()


class NGSSquare(NGSGeo):
    """Unit square with an inner rectangular subdomain.

    Boundary names standardised to match NGSSimple / unit_square convention:
      bottom | right | top | left   (outer edges)
      interface                      (inner subdomain boundary)
    Material names: "inner", "outer".
    """
    def __init__(self, size=0.3, **kwargs):
        super().__init__(**kwargs)
        assert size > 0, "Size of subdomain must be positive!"
        assert size < 1, "Size of subdomain must be <1!"

        outer = Rectangle(1, 1).Face()
        outer.edges.name = "outer"
        outer.edges.Max(X).name = "right"
        outer.edges.Min(X).name = "left"
        outer.edges.Min(Y).name = "bottom"
        outer.edges.Max(Y).name = "top"

        pos = (1 - size)/2
        inner = MoveTo(pos, pos).Rectangle(size, size).Face()
        inner.edges.name = "interface"

        outer = outer - inner
        inner.faces.name = "inner"
        inner.faces.col = (1, 0, 0)
        outer.faces.name = "outer"

        self.outer_bnd = "left|right|bottom|top"

        shape = Glue([inner, outer])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()

class NGSSquareHole(NGSGeo):
    """Unit square with a cross-shaped hole and a small circular inner subdomain.

    Boundary names standardised to match NGSSimple / unit_square convention:
      bottom | right | top | left   (outer edges)
      interface                      (inner subdomain boundary)
    Material names: "inner", "outer".
    """
    def __init__(self, size=0.1, channel_size=0.03, gap=0.15, **kwargs):
        super().__init__(**kwargs)
        assert size > 0, "Size of subdomain must be positive!"
        assert size < 1, "Size of subdomain must be <1!"

        outer = Rectangle(1, 1).Face()
        outer.edges.name = "outer"
        outer.edges.Max(X).name = "right"
        outer.edges.Min(X).name = "left"
        outer.edges.Min(Y).name = "bottom"
        outer.edges.Max(Y).name = "top"

        block_size = 1-2*gap
        block = MoveTo(gap,gap).Rectangle(block_size,block_size).Face()
        channel_h = MoveTo(gap,1/2-channel_size/2).Rectangle(block_size,channel_size).Face()
        channel_v = MoveTo(1/2-channel_size/2,gap).Rectangle(channel_size,block_size).Face()
        block = block - channel_h - channel_v
        
        self.centre = (1/2,1/2)
        inner = WorkPlane().Circle(self.centre[0], self.centre[1], size).Face()
        inner.edges.name = "interface"

        outer = outer - block - inner
        inner.faces.name = "inner"
        outer.faces.name = "outer"

        self.outer_bnd = "left|right|bottom|top"

        shape = Glue([inner, outer])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()

class NGSTriples(NGSGeo):
    """Unit square with three inner rectangular subdomains.

    Boundary names standardised to match NGSSimple / unit_square convention:
      bottom | right | top | left   (outer edges)
      interface                      (inner subdomain boundary)
    Material names: "inner", "outer".
    """
    def __init__(self, size=0.05, **kwargs):
        super().__init__(**kwargs)
        assert size > 0, "Size of subdomain must be positive!"
        assert size < 1, "Size of subdomain must be <1!"

        outer = Rectangle(2, 1).Face()
        outer.edges.name = "outer"
        outer.edges.Max(X).name = "right"
        outer.edges.Min(X).name = "left"
        outer.edges.Min(Y).name = "bottom"
        outer.edges.Max(Y).name = "top"

        self.centre = []

        cx = 0.25
        cy = 0.5
        inner1 = MoveTo(cx - size/2, cy - size/2).Rectangle(size, size).Face()
        inner1.edges.name = "interface1"
        self.centre.append([cx,cy])

        cx = 1.75
        cy = 0.75
        inner2 = MoveTo(cx - size/2, cy - size/2).Rectangle(size, size).Face()
        inner2.edges.name = "interface2"
        self.centre.append([cx,cy])

        cx = 1.75
        cy = 0.25
        inner3 = MoveTo(cx - size/2, cy - size/2).Rectangle(size, size).Face()
        inner3.edges.name = "interface3"
        self.centre.append([cx,cy])

        outer = outer - inner1 - inner2 - inner3
        inner1.faces.name = "inner1"
        inner2.faces.name = "inner2"
        inner3.faces.name = "inner3"
        outer.faces.name = "outer"

        self.outer_bnd = "left|right|bottom|top"

        shape = Glue([inner1, inner2, inner3, outer])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()

class NGSLShaped(NGSGeo):
    """L-shaped domain [0,1]^2 \\ [1/2,1]^2, with a small square inner subdomain
    near the re-entrant corner at (1/2,1/2).
    BC: outer (all exterior edges), interface (inner subdomain boundary).
    Material: 'inner', 'outer'.
    Classic benchmark for corner singularities (u ~ r^{2/3}).
    """
    def __init__(self, size=0.3, **kwargs):
        super().__init__(**kwargs)
        centre = 1/2
        assert 0 < size and size < centre
        big   = MoveTo(0, 0).Rectangle(1, 1).Face()
        notch = MoveTo(centre, centre).Rectangle(centre, centre).Face()
        domain = big - notch
        domain.edges.name = "outer"

        # Inner subdomain: small square just inside the re-entrant corner
        inner = MoveTo(centre-size, centre-size).Rectangle(size, size).Face()
        inner.edges.name = "interface"
        inner.faces.name = "inner"
        self.centre = (centre - size/2, centre - size/2)

        domain = domain - inner
        domain.faces.name = "outer"

        self.outer_bnd = "outer"
        shape = Glue([inner, domain])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()

class NGSBentChannel(NGSGeo):
    """90° bent channel (pipe elbow). Inner subdomain is a small arc-sector
    strip near the inner wall of the bend — the physically active region
    (thin reaction zone, heater strip, etc.).
    BC: wall (outer & inner walls), inlet, outlet, interface (inner subdomain).
    """
    def __init__(self, R=1.0, r=0.4, w=0.15, **kwargs):
        super().__init__(**kwargs)
        # Full annular quarter-disk
        outer_disk = WorkPlane().Circle(0, 0, R).Face()
        inner_disk  = WorkPlane().Circle(0, 0, r).Face()
        annulus = outer_disk - inner_disk

        # Restrict to first quadrant: subtract x<0 and y<0 half-planes
        big = 2 * R + 1
        left_half   = MoveTo(-big, -big).Rectangle(big, 2*big).Face()
        bottom_half = MoveTo(-big, -big).Rectangle(2*big, big).Face()
        channel = annulus - left_half - bottom_half

        channel.edges.name = "wall"
        channel.edges.Min(X).name = "inlet"
        channel.edges.Min(Y).name = "outlet"

        # Inner subdomain: thin arc strip just outside the inner wall
        inner_band_outer = WorkPlane().Circle(0, 0, r + w).Face()
        inner_band_inner = WorkPlane().Circle(0, 0, r).Face()
        inner_band = inner_band_outer - inner_band_inner
        inner_band = inner_band - left_half - bottom_half

        inner_band.edges.name = "interface"
        inner_band.faces.name = "inner"

        channel = channel - inner_band
        channel.faces.name = "outer"

        self.outer_bnd = "inlet|outlet|wall"
        shape = Glue([inner_band, channel])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()

class NGSPerforated(NGSGeo):
    """Unit square with a 3×3 grid of circular holes.
    Inner subdomain: a small square in the centre of the matrix (between holes).
    BC: outer (square boundary), holes (all hole boundaries), interface (inner subdomain).
    Material: 'inner', 'matrix'.
    """
    def __init__(self, n=3, hole_r=0.03, size=0.12, **kwargs):
        super().__init__(**kwargs)
        domain = Rectangle(1, 1).Face()
        
        domain.edges.name = "outer"
        domain.edges.Max(X).name = "right"
        domain.edges.Min(X).name = "left"
        domain.edges.Min(Y).name = "bottom"
        domain.edges.Max(Y).name = "top"

        # Inner subdomain: centred in the matrix, halfway between two hole rows
        #cx = 0.2 - size / 2
        #cy = 0.3 - size / 2
        #inner = MoveTo(cx, cy).Rectangle(size, size).Face()
        #inner.edges.name = "interface"
        #inner.faces.name = "inner"
        self.centre = None #(cx, cy)
        #domain = domain - inner

        x_offset = 0.5
        y_offset = 0.4
        gap = 0.1
        for x, y in [
                    [0,0],
                    #[1,1],
                    [2,2],
                    #[3,3],
                    #[4,4],
                    [-1,1],
                    #[-2,2],
                    [-3,3],
                    #[-4,4],
                    #[0,2],
                    #[-1,3],
                    #[1,3],
                    [-2,4],
                    #[0,4],
                    #[2,4]
                    ]:
            h = WorkPlane().Circle(x_offset + x*gap, y_offset + y*gap, hole_r).Face()
            h.edges.name = "scatter"
            domain = domain - h

        domain = domain.Rotate(Axis((0.5, 0.5, 0), Z), 45)
        self.corners = (
                           (0.5, 0.5 - 1/np.sqrt(2)),
                           (0.5 - 1/np.sqrt(2), 0.5),
                           (0.5, 0.5 + 1/np.sqrt(2)),
                           (0.5 + 1/np.sqrt(2), 0.5),
                       )
        self.outer_bnd = "left|right|bottom|top"
        self.geo = OCCGeometry(domain, dim=2)
        #self.geo = OCCGeometry(Glue([inner,domain]), dim=2)
        self.make()

class NGSGappy(NGSGeo):
    """Description
    """
    def __init__(self, n=3, hole_r=0.35, gap=0.05, size=0.1, **kwargs):
        super().__init__(**kwargs)
        domain = Rectangle(1, 1).Face()
        self.corners = ((0,0), (0,1), (1,1), (1,0))
        domain.edges.Max(X).name = "right"
        domain.edges.Min(X).name = "left"
        domain.edges.Min(Y).name = "bottom"
        domain.edges.Max(Y).name = "top"

        # Inner subdomain: centred in the matrix, halfway between two hole rows
        cx = 0.075
        cy = 0.5
        inner = MoveTo(cx - size / 2, cy - size / 2).Rectangle(size, size).Face()
        self.centre = (cx, cy)

        for direction in [1, -1]:
            h = WorkPlane().Circle(0.5, 0.5 + direction * gap / 2, hole_r).Face()
            s = MoveTo(0, 1/4 - direction * 1/4  + direction * gap / 2).Rectangle(1, 0.5).Face()
            h = h - s
            h.edges.name = "scatter"
            domain = domain - h
            inner = inner - h
        domain = domain - inner

        inner.edges.name = "interface"
        inner.faces.name = "inner"
        domain.edges.name = "outer"

        self.outer_bnd = "left|right|bottom|top"
        shape = Glue([inner,domain])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()

class NGSDoubleSlit(NGSGeo):
    """Rectangular domain with a thin wall containing two slits.
    Inner subdomain: small rectangle just upstream (left) of the wall —
    the 'source region' for the incident wave or diffusion flux.
    BC: inlet, outlet, wall, interface.
    """
    def __init__(self, L=2, H=1, wall_t=0.05, slit_w=0.05, gap=0.2, size=0.3, inner_gap=0.1, **kwargs):
        super().__init__(**kwargs)
        domain = MoveTo(0, 0).Rectangle(L, H).Face()

        # Wall with two slits punched out
        wall_full = MoveTo(L/2 - wall_t/2, 0).Rectangle(wall_t, H).Face()
        slit1 = MoveTo(L/2 - wall_t/2, H/2 - gap - slit_w).Rectangle(wall_t, slit_w).Face()
        slit2 = MoveTo(L/2 - wall_t/2, H/2 + gap).Rectangle(wall_t, slit_w).Face()
        wall = wall_full - slit1 - slit2
        wall.edges.name = "wall"
        domain = domain - wall

        # Inner subdomain: just upstream of the wall, centred vertically
        inner_x1 = L/2 - wall_t/2 - size - inner_gap
        inner_y = (H - size)/2
        inner1 = MoveTo(inner_x1, inner_y).Rectangle(size, size).Face()
        inner1.edges.name = "interface1"
        inner1.faces.name = "inner1"

        inner_x2 = L/2 + wall_t/2 + inner_gap
        inner2 = MoveTo(inner_x2, inner_y).Rectangle(size, size).Face()
        inner2.edges.name = "interface2"
        inner2.faces.name = "inner2"
        self.centre = [(inner_x1 + size/2, inner_y + size/2), (inner_x2 + size/2, inner_y + size/2)]

        domain = domain - inner1 - inner2
        domain.edges.Min(X).name = "left"
        domain.edges.Max(X).name = "right"
        domain.faces.name = "outer"

        self.outer_bnd = "left|right"
        shape = Glue([inner1, inner2, domain])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()

class NGSEllipseInPolygon(NGSGeo):
    """Irregular convex pentagon with an off-centre elliptic inclusion
    and a small circular inner subdomain (source region) on the opposite side.
    BC: outer (pentagon edges), interface_ellipse (ellipse), interface (source circle).
    Material: 'ellipse', 'inner', 'outer'.
    """
    def __init__(self, size=0.18, **kwargs):
        super().__init__(**kwargs)
        # Irregular convex pentagon
        verts = [(0.0, 0.0), (1.2, 0.05), (1.5, 0.8), (0.9, 1.4), (0.1, 1.1)]
        wp = WorkPlane()
        wp.MoveTo(*verts[0])
        for v in verts[1:]:
            wp.LineTo(*v)
        wp.Close()
        outer = wp.Face()
        outer.edges.name = "outer"

        # Elliptic inclusion (approximated as scaled circle via OCC transform)
        ellipse = WorkPlane().Circle(0.9, 0.85, 0.22).Face()
        ellipse.edges.name = "interface_ellipse"
        ellipse.faces.name = "ellipse"

        # Inner source subdomain: small circle on the lower-left side
        inner = WorkPlane().Circle(0.35, 0.35, size).Face()
        inner.edges.name = "interface"
        inner.faces.name = "inner"

        outer = outer - ellipse - inner
        outer.faces.name = "outer"

        self.outer_bnd = "outer"
        shape = Glue([ellipse, inner, outer])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()

class NGSDoubleHalfPipes(NGSGeo):
    """McDonald's-M shaped domain: two upward-arching half-annuli joined at the centre.

    Geometry
    --------
    Each half-pipe is the upper half of an annulus, built by boolean subtraction
    of a lower half-plane from a full annulus.

    Left pipe centred at (-d, 0), right at (+d, 0), where d = r + size.

    The inner subdomain is the upper half-disk of radius `size` at the origin
    (clipped to y >= 0), filling the middle foot of the M.

    Parameters
    ----------
    R : float
        Outer radius of each half-annulus (default 1.0).
    r : float
        Inner radius of each half-annulus (default 0.4).  Must satisfy r < R.
    size : float
        Radius of the central source half-disk (default 0.2).

    Boundary names
    --------------
    left      : flat base edges (y=0) of the left half-pipe
    right     : flat base edges (y=0) of the right half-pipe
    wall      : all curved arcs of both half-annuli
    interface : boundary of the central source half-disk

    Material names
    --------------
    inner  : the central source half-disk
    outer  : the rest of the M-shaped domain

    outer_bnd = 'left|right|wall'
    """

    def __init__(self, R=1.0, r=0.4, size=0.35, **kwargs):
        super().__init__(**kwargs)
        assert r < R,    "Inner radius r must be < outer radius R."
        assert size > 0, "size must be positive."

        d = r + size   # half-distance between pipe centres

        # Clipping rectangle: removes the lower half-plane (y < 0)
        big = 2 * (R + d) + 2
        lower_half = MoveTo(-big, -big).Rectangle(2 * big, big).Face()

        def half_annulus(cx):
            """Upper half of an annulus centred at (cx, 0) via boolean clip."""
            outer_disk = WorkPlane().Circle(cx, 0, R).Face()
            inner_disk  = WorkPlane().Circle(cx, 0, r).Face()
            annulus = outer_disk - inner_disk
            return annulus - lower_half

        left_pipe  = half_annulus(-d)
        right_pipe = half_annulus(+d)

        # Inner subdomain: upper half-disk at origin (clipped to y >= 0)
        inner_full = WorkPlane().Circle(0, 0, size).Face()
        inner = inner_full - lower_half
        inner.edges.Max(Y).name = "interface"
        inner.faces.name = "inner"

        left_pipe = left_pipe - inner
        right_pipe = right_pipe - inner

        left_pipe.edges.name = "wall"
        left_pipe.edges.Min(X).name = "left"
        right_pipe.edges.name = "wall"
        right_pipe.edges.Max(X).name = "right"

        outer = left_pipe + right_pipe
        outer.faces.name = "outer"

        self.outer_bnd = "left|right"#|wall"
        shape = Glue([inner, outer])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()


class NGSCapillary(NGSGeo):
    """Irregular capillary network inside a unit circle, connected to a central hub.

    The domain consists of two subdomains:
      - 'inner'     : small circular hub at the origin
      - 'capillary' : a dense network of thin, kinked tubes radiating from the hub,
                      some reaching the outer circle (escape routes), others
                      terminating as dead ends or cross-linking internally.

    The outer matrix (space between capillaries) is NOT part of the domain.

    Boundary names
    --------------
    outer     : arcs where capillaries intersect the outer circle (escape exits)
    interface : where capillaries adjoin the hub
    wall      : all other capillary boundaries (dead ends, tube walls)

    Parameters
    ----------
    R_outer : float  Outer circle radius (default 1.0).
    R_inner : float  Hub radius (default 0.10).
    W       : float  Capillary half-width, total tube diameter = 2*W (default 0.018).
    seed    : int    RNG seed controlling kink placement (default 11).
    """

    # Centre-line endpoints for all capillary segments.
    # Segments whose far endpoint lies outside R_outer will be clipped to the circle.
    _SEGS = [
        # 8 primary spokes – hub to outer boundary
        ((0.00,  0.10), ( 0.03,  1.00)),
        ((0.00, -0.10), (-0.05, -1.00)),
        (( 0.10, 0.00), ( 1.00,  0.05)),
        ((-0.10, 0.00), (-1.00, -0.03)),
        (( 0.07,  0.07), ( 0.70,  0.72)),
        ((-0.07,  0.07), (-0.72,  0.69)),
        (( 0.07, -0.07), ( 0.71, -0.70)),
        ((-0.07, -0.07), (-0.69, -0.71)),
        # 8 secondary Y-forks branching off mid-spokes
        (( 0.30,  0.42), ( 0.55,  0.12)),
        (( 0.30,  0.42), ( 0.08,  0.62)),
        ((-0.28,  0.42), (-0.50,  0.12)),
        ((-0.28,  0.42), (-0.06,  0.60)),
        (( 0.40, -0.36), ( 0.72, -0.18)),
        (( 0.40, -0.36), ( 0.18, -0.72)),
        ((-0.38, -0.34), (-0.70, -0.14)),
        ((-0.38, -0.34), (-0.14, -0.70)),
        # 4 criss-crossing cross-links
        ((-0.58,  0.14), ( 0.58,  0.16)),
        (( 0.14, -0.58), ( 0.16,  0.58)),
        ((-0.48, -0.28), ( 0.44,  0.22)),
        (( 0.52,  0.32), (-0.18,  0.54)),
        # 4 dead-end stubs (terminate inside)
        (( 0.20,  0.50), ( 0.24,  0.78)),
        ((-0.16, -0.52), (-0.20, -0.78)),
        (( 0.76,  0.06), ( 0.58,  0.34)),
        ((-0.76, -0.04), (-0.58, -0.32)),
    ]

    def __init__(self, R_outer=1.0, R_inner=0.10, W=0.018, seed=11, **kwargs):
        super().__init__(**kwargs)
        rng = np.random.default_rng(seed)

        def kink(p0, p1, n_kinks=2, sigma=0.20):
            """Insert randomly displaced waypoints between p0 and p1."""
            p0, p1 = np.array(p0), np.array(p1)
            d = p1 - p0
            L = np.linalg.norm(d)
            if L < 1e-12:
                return [p0, p1]
            t_hat = d / L
            n_hat = np.array([-t_hat[1], t_hat[0]])
            ts = np.sort(rng.uniform(0.25, 0.75, n_kinks))
            pts = [p0]
            for t in ts:
                pts.append(p0 + t * d + rng.normal(0, sigma * L) * n_hat)
            pts.append(p1)
            return pts

        def tube_face(pts, half_w):
            """OCC face for a tube of width 2*half_w along a polyline."""
            face = None
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                dx, dy = x1 - x0, y1 - y0
                L = np.hypot(dx, dy)
                if L < 1e-10:
                    continue
                ux, uy = dx / L, dy / L
                nx, ny = -uy, ux
                c0m = (x0 - half_w * nx, y0 - half_w * ny)
                c0p = (x0 + half_w * nx, y0 + half_w * ny)
                c1p = (x1 + half_w * nx, y1 + half_w * ny)
                c1m = (x1 - half_w * nx, y1 - half_w * ny)
                wp = WorkPlane()
                wp.MoveTo(*c0m)
                wp.LineTo(*c1m)
                wp.LineTo(*c1p)
                wp.LineTo(*c0p)
                wp.Close()
                rect = wp.Face()
                cap0 = WorkPlane().Circle(x0, y0, half_w).Face()
                cap1 = WorkPlane().Circle(x1, y1, half_w).Face()
                seg = rect + cap0 + cap1
                face = seg if face is None else face + seg
            return face

        # Build tube union
        tube_union = None
        for (p0, p1) in self._SEGS:
            pts = kink(p0, p1, n_kinks=2, sigma=0.20)
            pts = [list(q) for q in pts]
            tf = tube_face(pts, W)
            tube_union = tf if tube_union is None else tube_union + tf

        # Named boundary faces – name outer circle BEFORE boolean so OCC
        # propagates "outer" to all clipped arcs automatically.
        outer_disk = WorkPlane().Circle(0, 0, R_outer).Face()
        outer_disk.edges.name = "outer"

        self.centre = [0, 0]
        hub = WorkPlane().Circle(self.centre[0], self.centre[1], R_inner).Face()
        hub.edges.name = "interface"
        hub.faces.name = "inner"
        
        # Domain = (tubes ∩ outer_disk) − hub
        capillary = tube_union * outer_disk - hub
        capillary.faces.name = "capillary"
        for e in capillary.edges:
            if e.name not in ("outer", "interface"):
                e.name = "wall"

        self.outer_bnd = "outer"
        shape = Glue([hub, capillary])
        self.geo = OCCGeometry(shape, dim=2)
        self.make()
