function flux(u, axis)
    w = zero(u)
    dif!(w, d, conj(u), u; axis)
    imag(w)
end

function rflux(u)
    (x.*flux(u,1) + y.*flux(u,2))./r
end