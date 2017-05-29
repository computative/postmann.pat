while (i < iterations ) {
    k = rand_particle(gen);
    not_k = (k+1) % 2;

    rijpp = norm(rpp.col(0)-rpp.col(1));
    Fpp = -2*a*w*rpp.col(k) + 2*c*(rpp.col(k) - rpp.col(not_k))/( (1 + b*rijpp)*(1 + b*rijpp)*rijpp );

    r.col(k) = rpp.col(k) + D*Fpp*dt + randn<vec>(2)*sqrt(dt);
    rij = norm(r.col(0)-r.col(1));
    F    = -2*a*w*r.col(k) + 2*c*(r.col(k) - r.col(not_k))/( (1 + b*rij)*(1 + b*rij)*rij );
    p = rpp.col(k) - r.col(k) - D*dt*F;
    q = r.col(k) - rpp.col(k) - D*dt*Fpp;
    qji = exp(- dot(p,p)/(4*D*dt));
    qij = exp(- dot(q,q)/(4*D*dt));
    wf = psi(r.col(0),r.col(1), a,b,c,w);

    if ( wf*wf*qji/(wfpp*wfpp*qij ) > rand_double(gen) ) {
        rpp = r; wfpp = wf; rij = norm(r.col(0) - r.col(1) );
    }
    e = 1/rij + 0.5*w*w*(1-a*a)*( dot(rpp.col(0),rpp.col(0)) + dot(rpp.col(1),rpp.col(1)) )
            + 2*a*w + a*w*c*rij/pow(1 + rij*b,2) - c*(1+rij*c-b*b*rij*rij)/( rij*pow(1 + rij*b,4) );
    E += e/iterations;
    i++;
}
