(define (domain dxpflc3up53obvi)
    (:requirements :strips)
    (:predicates (a46zhwykn6jvbabk ?airport)  (abwaom5mhyf979wg ?obj1 ?obj2)  (b6e9q4r60gagvdcn ?city)  (cql4o62p1yeke3ok ?airplane)  (dvr5qjofrygv9zlx ?obj ?loc)  (fe8cgp3y92p6lb85 ?loc)  (j3bxzst7l100cyne ?obj ?city)  (lwiiiuk750o3u28e ?obj)  (rvgqqfhogwl9yspo ?truck))
    (:action af9tse23ljclsqad
        :parameters (?ob_1 ?ob_2 ?ob_3 ?ob_4)
        :precondition (and (rvgqqfhogwl9yspo ?ob_1) (fe8cgp3y92p6lb85 ?ob_2) (fe8cgp3y92p6lb85 ?ob_3) (b6e9q4r60gagvdcn ?ob_4) (dvr5qjofrygv9zlx ?ob_1 ?ob_2) (j3bxzst7l100cyne ?ob_2 ?ob_4) (j3bxzst7l100cyne ?ob_3 ?ob_4))
        :effect (and (not (dvr5qjofrygv9zlx ?ob_1 ?ob_2)) (dvr5qjofrygv9zlx ?ob_1 ?ob_3))
    )
     (:action j4gv801gnu2it0yj
        :parameters (?ob_1 ?ob_2 ?ob_3)
        :precondition (and (lwiiiuk750o3u28e ?ob_1) (rvgqqfhogwl9yspo ?ob_2) (fe8cgp3y92p6lb85 ?ob_3) (dvr5qjofrygv9zlx ?ob_2 ?ob_3) (abwaom5mhyf979wg ?ob_1 ?ob_2))
        :effect (and (not (abwaom5mhyf979wg ?ob_1 ?ob_2)) (dvr5qjofrygv9zlx ?ob_1 ?ob_3))
    )
     (:action kip9uw781pv62umn
        :parameters (?ob_1 ?ob_2 ?ob_3)
        :precondition (and (lwiiiuk750o3u28e ?ob_1) (cql4o62p1yeke3ok ?ob_2) (fe8cgp3y92p6lb85 ?ob_3) (dvr5qjofrygv9zlx ?ob_1 ?ob_3) (dvr5qjofrygv9zlx ?ob_2 ?ob_3))
        :effect (and (not (dvr5qjofrygv9zlx ?ob_1 ?ob_3)) (abwaom5mhyf979wg ?ob_1 ?ob_2))
    )
     (:action tv30k33pzoulql6w
        :parameters (?ob_1 ?ob_2 ?ob_3)
        :precondition (and (cql4o62p1yeke3ok ?ob_1) (a46zhwykn6jvbabk ?ob_2) (a46zhwykn6jvbabk ?ob_3) (dvr5qjofrygv9zlx ?ob_1 ?ob_2))
        :effect (and (not (dvr5qjofrygv9zlx ?ob_1 ?ob_2)) (dvr5qjofrygv9zlx ?ob_1 ?ob_3))
    )
     (:action u64y1a9apusmslxb
        :parameters (?ob_1 ?ob_2 ?ob_3)
        :precondition (and (lwiiiuk750o3u28e ?ob_1) (rvgqqfhogwl9yspo ?ob_2) (fe8cgp3y92p6lb85 ?ob_3) (dvr5qjofrygv9zlx ?ob_2 ?ob_3) (dvr5qjofrygv9zlx ?ob_1 ?ob_3))
        :effect (and (not (dvr5qjofrygv9zlx ?ob_1 ?ob_3)) (abwaom5mhyf979wg ?ob_1 ?ob_2))
    )
     (:action wio5amhq7814n006
        :parameters (?ob_1 ?ob_2 ?ob_3)
        :precondition (and (lwiiiuk750o3u28e ?ob_1) (cql4o62p1yeke3ok ?ob_2) (fe8cgp3y92p6lb85 ?ob_3) (abwaom5mhyf979wg ?ob_1 ?ob_2) (dvr5qjofrygv9zlx ?ob_2 ?ob_3))
        :effect (and (not (abwaom5mhyf979wg ?ob_1 ?ob_2)) (dvr5qjofrygv9zlx ?ob_1 ?ob_3))
    )
)

