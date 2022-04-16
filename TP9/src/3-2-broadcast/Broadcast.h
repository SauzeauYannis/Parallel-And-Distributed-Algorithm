#pragma once

/* Effectuer une diffusion en pipeline ...  */
void Broadcast(
    const int k, // numéro du processeur émetteur
    int *const addr, // entiers à envoyer
    const int N, // nombre d'entiers à envoyer au total
    const int M // packet size
);
