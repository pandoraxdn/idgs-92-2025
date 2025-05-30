import { useState } from 'react';

interface UseContadorBasico{
    contador: number;
    add: () => void;
    dec: () => void;
    reset: () => void;
}

export const useContadorBasicoHook = (): UseContadorBasico => {

    const initialState: number = 10;

    const [ contador, setContador ] = useState<number>(initialState);

    const add = () => setContador(contador + 1);
    const dec = () => setContador((contador == 0) ? 0 : contador - 1);
    const reset = () => setContador(initialState);

    return {
        contador,
        add,
        dec,
        reset
    }
}
