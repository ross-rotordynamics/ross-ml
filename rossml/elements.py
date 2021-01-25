"""Bearing and seal elements module.

This module defines bearing and seal elements through artificial neural networks (ANN).
There is a general class to create an element with any ANN built with ROSS-ML and
other specific classes, which have fixed features that the model must match.
"""

from collections.abc import Iterable
from itertools import repeat

import numpy as np
import pandas as pd
from ross import BearingElement

from rossml.pipeline import Model

__all__ = ["AnnBearingElement", "SealLabyrinthElement", "TiltPadBearingElement"]


class AnnBearingElement(BearingElement):
    """Create bearings or seals via Neural Netwarks.

    This class creates a element from a trained neural network. In order to create it
    properly, one should bear in mind the name of the trained network, and the
    parameters used. The results will be given in a dataframe form. When the number of
    features is different from the trained one, a error is displayed and the same
    occurs when the variable name is wrong due to a typo.

    Parameters
    ----------
    arq : str
        The neural network folder's name, which must be located at ross-ml package.
        The model files are loaded from this folder.
    n : int
        The node in which the element will be located in the rotor.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.
    kwargs : optional
        The required arguments to the neural network predict the rotordynamic
        coefficients. It must match the features from the neural network.
        It may varies with the loaded model.

    Returns
    -------
    A AnnBearingElement object.

    Raises
    ------
    KeyError
        Error raised if kwargs does not match the features from the neural netowork
        model.
    ValueError
        Error raised if some kwargs does not have the same length than 'speeds' kwargs.
        It is raised only if other kwargs rathen than 'speeds' is passed as an iterable,
        with missmatching size.

    Examples
    --------
    >>> import rossml as rsml

    Specify the neural netowark to be used. "arq" must match one of the folders name
    of a neural network previously saved inside rossml. You can check the available
    models with:

    >>> rsml.available_models()
    ['test_model']

    Now, select one of the available options.
    >>> nn_model = "test_model"

    Or build a neu neural network model (see Pipeline documentation).

    Before setting data to the neural network, check what are the required features
    according to the loaded model.

    First, load the neural network:

    >>> model = rsml.Model(nn_model)

    Check for the features:
    >>> features = model.features

    Now, enter the respective values for each key in "model.features". Or use other
    classes with preset parameters.

    >>> seal = rsml.AnnBearingElement(
    ...     n=0,
    ...     arq=nn_model,
    ...     seal_radius=141.61,
    ...     number_of_teeth=22,
    ...     tooth_pitch=8.58,
    ...     tooth_height=8.37,
    ...     radial_clearance=0.16309,
    ...     ethane=0.14548,
    ...     propane=0.00542,
    ...     isobutan=.17441,
    ...     butane=0.22031,
    ...     nitrogen=0.10908,
    ...     methane=0.11907,
    ...     hydrogen=0.11584,
    ...     oxygen=0.05794,
    ...     co2=0.05224,
    ...     reservoir_temperature=25.0,
    ...     reservoir_pressure=568.90,
    ...     sump_pressure=5.3,
    ...     inlet_tangential_velocity_ratio=0.617,
    ...     whirl_speed=8310.5,
    ...     speeds=[7658.3],
    ... )
    >>> seal # doctest: +ELLIPSIS
    AnnBearingElement(n=0...

    If kwargs has different argumments than model.features, an error is raised
    informing how many which are the features and how many kwargs has been entered.
    Users can copy the list directly from the error message to set the correct keys in
    kwargs.

    >>> seal = rsml.AnnBearingElement(
    ...     n=0,
    ...     arq=nn_model,
    ...     seal_radius=141.61,
    ...     number_of_teeth=22,
    ...     tooth_pitch=8.58,
    ...     tooth_height=8.37,
    ...     radial_clearance=0.16309,
    ...     methane=0.11907,
    ...     hydrogen=0.11584,
    ...     oxygen=0.05794,
    ...     co2=0.05224,
    ...     reservoir_temperature=25.0,
    ...     reservoir_pressure=568.90,
    ...     sump_pressure=5.3,
    ...     inlet_tangential_velocity_ratio=0.617,
    ...     whirl_speed=8310.5,
    ...     speeds=[7658.3],
    ... ) # doctest: +ELLIPSIS
    KeyError...
    """

    def __init__(self, arq=None, n=None, n_link=None, scale_factor=1.0, **kwargs):
        # loading neural network model
        model = Model(arq)
        features = model.features
        reordered_dict = {}

        # checking data consistency
        if any(key not in kwargs.keys() for key in features):
            raise KeyError(
                f"Model '{arq}' has the following {len(list(features))} features: "
                f"{list(features)}, and {len(kwargs)} are given. "
                f"Check the **kwargs dictionary for the same keys."
            )

        size = len(kwargs["speeds"])

        if any(len(v) != size for k, v in kwargs.items() if isinstance(v, Iterable)):
            raise ValueError(
                "Some keyword arguments does not have the same length than 'speeds'."
            )

        for k in kwargs:
            if not isinstance(kwargs[k], Iterable):
                reordered_dict[k] = repeat(kwargs[k], size)
            else:
                reordered_dict[k] = kwargs[k]

        reordered_dict = {k: kwargs[k] for k in features}
        data = pd.DataFrame(reordered_dict)
        results = model.predict(data)

        super().__init__(
            n=n,
            frequency=np.array(kwargs["speeds"]),
            kxx=np.array(results["kxx"], dtype=np.float64),
            kxy=np.array(results["kxy"], dtype=np.float64),
            kyx=np.array(results["kyx"], dtype=np.float64),
            kyy=np.array(results["kyy"], dtype=np.float64),
            cxx=np.array(results["cxx"], dtype=np.float64),
            cxy=np.array(results["cyx"], dtype=np.float64),
            cyx=np.array(results["cyx"], dtype=np.float64),
            cyy=np.array(results["cyy"], dtype=np.float64),
            tag=arq,
            n_link=n_link,
            scale_factor=scale_factor,
        )


class SealLabyrinthElement(AnnBearingElement):
    """Create labyrinth Seal elements via Neural Netwarks.

    This class creates an Labyrinth Seal Element from a trained neural network.
    The parameters inserted are used to predict the rotordynamics coefficients.

    Parameters
    ----------
    n : int
        The node in which the element will be located in the rotor.
    arq : str
        The neural network folder's name, which must be located at ross-ml package.
        The model files are loaded from this folder.
    seal_radius : float, list, array
        The seal radius.
    number_of_teeth : int, list, array
        Number of teeth present on the seal .
    tooth_pitch : float, list, array
        The pitch between two teeth.
    tooth_height : float, list, array
        The tooth height.
    radial_clearance : float, list, array
        The seal radial clearance.
    methane : float, list, array
        The proportion of methane in the gas.
    ethane : float, list, array
        The proportion of ethane in the gas.
    propane : float, list, array
        The proportion of propane in the gas.
    isobutan : float, list, array
        The proportion of isobutan in the gas.
    butane : float, list, array
        The proportion of butane in the gas.
    hydrogen : float, list, array
        The proportion of hydrogen in the gas.
    nitrogen : float, list, array
        The proportion of nitrogen in the gas.
    oxygen : float, list, array
        The proportion of oxygen in the gas.
    co2 : float, list, array
        The proportion of co2 in the gas.
    reservoir_temperature : float, list, array
        The reservoir temperature.
    reservoir_pressure : float, list, array
        The reservoir pressure.
    sump_pressure : float, list, array
        The sump pressure.
    inlet_tangential_velocity_ratio, : float, list, array
        The inlet tangential velocity ratio.
    whirl_speed : float, list, array
        Whirl speed value.
    speeds : list, array
        Array with the frequencies of interest.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.

    Returns
    -------
    A SealLabyrinthElement object.

    Examples
    --------
    >>> import rossml as rsml
    >>> seal = rsml.SealLabyrinthElement(
    ...     arq="test_model", n=0,
    ...     seal_radius=141.60965632804,
    ...     number_of_teeth=22,
    ...     tooth_pitch=8.58370220849736,
    ...     tooth_height=8.369750873237724,
    ...     radial_clearance=0.16309012139139262,
    ...     methane=0.11907483818379025,
    ...     hydrogen=0.11584137533357493,
    ...     oxygen=0.05794358525114542,
    ...     co2=0.052243764797109724,
    ...     ethane=0.1454836854619626,
    ...     propane=0.005424100482971933,
    ...     isobutan=0.17441663090080106,
    ...     butane=0.22031291584401053,
    ...     nitrogen=0.10908265522969644,
    ...     reservoir_temperature=25.037483348934998,
    ...     reservoir_pressure=568.9058098384347,
    ...     sump_pressure=5.299447680455862,
    ...     inlet_tangential_velocity_ratio=0.6171344358228346,
    ...     whirl_speed=8310.497837226783,
    ...     speeds=[7658.340362809778],
    ... )
    >>> seal # doctest: +ELLIPSIS
    SealLabyrinthElement(n=0...
    """

    def __init__(
        self,
        n,
        arq,
        seal_radius,
        number_of_teeth,
        tooth_pitch,
        tooth_height,
        radial_clearance,
        methane,
        ethane,
        propane,
        isobutan,
        butane,
        hydrogen,
        nitrogen,
        oxygen,
        co2,
        reservoir_temperature,
        reservoir_pressure,
        sump_pressure,
        inlet_tangential_velocity_ratio,
        whirl_speed,
        speeds,
        n_link=None,
        scale_factor=1.0,
    ):

        super().__init__(
            arq=arq,
            n=n,
            n_link=n_link,
            scale_factor=scale_factor,
            seal_radius=seal_radius,
            number_of_teeth=number_of_teeth,
            tooth_pitch=tooth_pitch,
            tooth_height=tooth_height,
            radial_clearance=radial_clearance,
            methane=methane,
            ethane=ethane,
            propane=propane,
            isobutan=isobutan,
            butane=butane,
            hydrogen=hydrogen,
            nitrogen=nitrogen,
            oxygen=oxygen,
            co2=co2,
            reservoir_temperature=reservoir_temperature,
            reservoir_pressure=reservoir_pressure,
            sump_pressure=sump_pressure,
            inlet_tangential_velocity_ratio=inlet_tangential_velocity_ratio,
            whirl_speed=whirl_speed,
            speeds=speeds,
        )


class TiltPadBearingElement(AnnBearingElement):
    """Create Tilting Pad Bearing Element via Neural Netwarks.

    This class creates a Tilting Pad Bearing Element from a trained neural network.
    The parameters inserted are used to predict the rotordynamics coefficients.

    Parameters
    ----------
    n : int
        The node in which the element will be located in the rotor.
    arq : str
        The neural network folder's name, which must be located at ross-ml package.
        The model files are loaded from this folder.
    diameter : float
        Rotor diameter.
    axial_length : float
        Axial length.
    number_of_pads : int
        Number of pads.
    pad_leading_edge : float
        Angular position of first pad leading edge, relative to negative X axis.
    pad_thickness : float
        Pad thickness.
    ambient_pressure : float
        Ambient pressure
    supply_pressure : float
        Oil supply pressure.
    cavitation_pressure : float
        Cavitation pressure.
    supply_temperature : float
        Oil supply temperature.
    viscosity_at_supply : float
        Oil viscosity at supply temperature.
    density : float
        Oil density.
    specific_heat : float
        Oil specific heat
    thermal_conductivity : float
        Oil thermal conductivity.
    alpha_v : float
        Oil viscosity exponent
        visc = visc_supply * exp(-alpha_v * (T - T_supply))
    inertia_effects : boolen
        Key to considerer or not the inertia effects.
    frequency_analysis_option : int
        The frequency analysis type.
        Options are:
            1: synchronous
            2: asynchronous.
    shaft_speed : float
        shaft speed for asynchronous analysis only .
    thermal_analysis_type : int
        Options are 1 to 7.
        See THD equations for reference. Haussen (4) recommended.
    journal_heat_transfer : int
        Heat transfer analysis type for the journal bearing:
            1: Adiabatic journal.
            2: Known journal temperature.
            3: Calc journal temperature.
    journal_temperature : float
        Journal temperature if journal_heat_transfer == 2.
    pad_heat_transfer : int
        Heat transfer analysis type for the pad:
            1: Adiabatic pad.
            2: Known pad temperature.
            3: Calc pad temperature.
    sump_temperature : float
        Pad temperature if pad_heat_transfer == 2.
    pad_thermal_conductivity : float
        The pad thermal conductivity property.
    reynolds_pad_back : float
    housing_inner_diameter : float
        The housing inned diameter. It must be >= diameter + pad_thickness.
    percent_heat_in_oil : float
        Percentage of heat carried by the oil.
    groove_convection_coef : float
        The groove convection coefficient.
    inlet_taper_angle : float
        Pad inlet taper angle.
    inlet_taper_arc_length : float
        (degrees)
    case_analysis_option : int
        Options are:
            1: vary eccentricity
            2: vary load
    x_initial_eccentricity : float
        X eccentricity ratio initial guess. X0/cp.
    y_initial_eccentricity : float
        Y eccentricity ratio initial guess. Y0/cp.
    pad_option : int
        Options are:
            1: equal pads
            2: unequal pads
    analysis_model : int
        Thermal analysis type. Options are:
            1: TEHD
            2: THD
    pad_preload : float, list
        Pad preload. May be list if pad_option == 2
    pad_offset : float, list
        Pad offset. May be list if pad_option == 2
    pad_arc_length : float, list
        Pad arc length (degrees). May be list if pad_option == 2.
    pad_clearance : float, list
        Pad clearance. May be list if pad_option == 2.
    bearing_type : int
        Options are:
            1: tilting pad
            2: rigid pad
    pad_mass : float
        Pad mass (kg).
    pad_moment_of_inertia : flaot
        Moment of inertia about pivot (kg-m^2).
    pad_mass_center_location : float
        Distance from pivot do CM (m).
    pad_thickness_at_pivot : float
        Pad thickness at the pivot (m).
    pivot_type
    pivot_info : list, array
        4x3 list indicating pivot parameters for chosen pad_type.
    groove_mixing_model : int
        Options are:
            1: hot oil carry over
            2: improved model
    hot_oil_carry_over : float
        Hot oil carry over. Insert this option if groove_mixing_model = 1.
    groove_mixing_efficiency : float
        Groove mixing efficiency. Insert this option if groove_mixing_model = 2.
    oil_flowrate : float
        Oil flowrate (LPM). Insert this option if groove_mixing_model = 2.
    pad_deformation_model : int
        Pad deformation type. Options are:
            1: pressure
            2: thermal
            3: both
    shaft_and_housing : int, optional
        Options are:
            1: rigid shaft + housing
            2: shaft expands
            3: both expand
            4: shaft expands housing contracts
    temperature_cold_condition : float, optional
        Ttemperature cold condition (degC)
    shaft_thermal_expansion_coef : float, optional
        shaft_thermal_expansion_coef (1/degC). Default is None.
    pad_thermal_expansion_coef : float, optional
        pad_thermal_expansion_coef (1/degC). Default is None.
    housing_thermal_expansion_coef : float, optional
        Housing thermal expansion coefficient (1/degC). Default is None.
    pad_flexibility : int
        Options are:
            1: approximate
            2: 3D FEM)
    pad_elastic_modulus : float, optional
        Pad elastic modulus (N/m^2). Default is None
    pad_liner_compliance : int, optional
        Options are:
            1: liner deformation
            2: rigid liner
    liner_elastic_modulus : float, optional
        Linear elastic modulus (N/m^2). Defaults is None.
    liner_poisson_ratio : float
        Poisson ratio.. Defaults is None.
    liner_thickness : float, optional
        Linear thickness (m). Defaults is None.
    liner_thermal_expansion_coef : float, optional
        Liner thermal expansion coefficient (1/degC). Defaults is None.
    liner_thermal_conductivity : float, optional
        Liner thermal conductivity (W/m-degC). Defaults is None.
    grid_ratio : float, optional
        mesh ratio circ/axial. Defaults is None.
    number_of_circumferential_points : int, optional
        Number of circumferential points. Default is None.
    number_of_axial_points : int
        Number of axial points
    bearing_load_x : float, optional
        Bearing load X direction (N).
    bearing_load_y : float, optional
        Bearing load Y direction (N) (gravity load is negative).
    speeds : list, array
        Array with the frequencies of interest.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.

    Returns
    -------
    A TiltPadBearingElement object.
    """

    def __init__(
        self,
        n,
        arq,
        diameter,
        axial_length,
        number_of_pads,
        pad_leading_edge,
        pad_thickness,
        ambient_pressure,
        supply_pressure,
        cavitation_pressure,
        supply_temperature,
        viscosity_at_supply,
        density,
        specific_heat,
        thermal_condutivity,
        alpha_v,
        inertia_effects,
        frequency_analysis_option,
        shaft_speed,
        thermal_analysis_type,
        journal_heat_transfer,
        journal_temperature,
        pad_heat_transfer,
        sump_temperature,
        pad_thermal_conductivity,
        reynolds_pad_back,
        housing_inner_diameter,
        percent_heat_in_oil,
        groove_convection_coef,
        inlet_taper_angle,
        inlet_taper_arc_length,
        case_analysis_option,
        x_initial_eccentricity,
        y_initial_eccentricity,
        pad_option,
        analysis_model,
        pad_preload,
        pad_offset,
        pad_arc_length,
        pad_clearance,
        bearing_type,
        pad_mass,
        pad_moment_of_inertia,
        pad_mass_center_location,
        pad_thickness_at_pivot,
        pivot_type,
        pivot_info,
        groove_mixing_model,
        hot_oil_carry_over,
        groove_mixing_efficiency,
        oil_flowrate,
        pad_deformation_model,
        shaft_and_housing,
        temperature_cold_condition,
        shaft_thermal_expansion_coef,
        pad_thermal_expansion_coef,
        housing_thermal_expansion_coef,
        pad_flexibility,
        pad_elastic_modulus,
        pad_liner_compliance,
        liner_elastic_modulus,
        liner_poisson_ratio,
        liner_thickness,
        liner_thermal_expansion_coef,
        liner_thermal_conductivity,
        grid_ratio,
        number_of_circumferential_points,
        number_of_axial_points,
        bearing_load_x,
        bearing_load_y,
        speeds=None,
        n_link=None,
        scale_factor=1.0,
    ):

        super().__init__(
            arq=arq,
            n=n,
            n_link=n_link,
            scale_factor=scale_factor,
            diameter=diameter,
            axial_length=axial_length,
            number_of_pads=number_of_pads,
            pad_leading_edge=pad_leading_edge,
            pad_thickness=pad_thickness,
            ambient_pressure=ambient_pressure,
            supply_pressure=supply_pressure,
            cavitation_pressure=cavitation_pressure,
            supply_temperature=supply_temperature,
            viscosity_at_supply=viscosity_at_supply,
            density=density,
            specific_heat=specific_heat,
            thermal_condutivity=thermal_condutivity,
            alpha_v=alpha_v,
            inertia_effects=inertia_effects,
            frequency_analysis_option=frequency_analysis_option,
            shaft_speed=shaft_speed,
            thermal_analysis_type=thermal_analysis_type,
            journal_heat_transfer=journal_heat_transfer,
            journal_temperature=journal_temperature,
            pad_heat_transfer=pad_heat_transfer,
            sump_temperature=sump_temperature,
            pad_thermal_conductivity=pad_thermal_conductivity,
            reynolds_pad_back=reynolds_pad_back,
            housing_inner_diameter=housing_inner_diameter,
            percent_heat_in_oil=percent_heat_in_oil,
            groove_convection_coef=groove_convection_coef,
            inlet_taper_angle=inlet_taper_angle,
            inlet_taper_arc_length=inlet_taper_arc_length,
            case_analysis_option=case_analysis_option,
            x_initial_eccentricity=x_initial_eccentricity,
            y_initial_eccentricity=y_initial_eccentricity,
            pad_option=pad_option,
            analysis_model=analysis_model,
            pad_preload=pad_preload,
            pad_offset=pad_offset,
            pad_arc_length=pad_arc_length,
            pad_clearance=pad_clearance,
            bearing_type=bearing_type,
            pad_mass=pad_mass,
            pad_moment_of_inertia=pad_moment_of_inertia,
            pad_mass_center_location=pad_mass_center_location,
            pad_thickness_at_pivot=pad_thickness_at_pivot,
            pivot_type=pivot_type,
            pivot_info=pivot_info,
            groove_mixing_model=groove_mixing_model,
            hot_oil_carry_over=hot_oil_carry_over,
            groove_mixing_efficiency=groove_mixing_efficiency,
            oil_flowrate=oil_flowrate,
            pad_deformation_model=pad_deformation_model,
            shaft_and_housing=shaft_and_housing,
            temperature_cold_condition=temperature_cold_condition,
            shaft_thermal_expansion_coef=shaft_thermal_expansion_coef,
            pad_thermal_expansion_coef=pad_thermal_expansion_coef,
            housing_thermal_expansion_coef=housing_thermal_expansion_coef,
            pad_flexibility=pad_flexibility,
            pad_elastic_modulus=pad_elastic_modulus,
            pad_liner_compliance=pad_liner_compliance,
            liner_elastic_modulus=liner_elastic_modulus,
            liner_poisson_ratio=liner_poisson_ratio,
            liner_thickness=liner_thickness,
            liner_thermal_expansion_coef=liner_thermal_expansion_coef,
            liner_thermal_conductivity=liner_thermal_conductivity,
            grid_ratio=grid_ratio,
            number_of_circumferential_points=number_of_circumferential_points,
            number_of_axial_points=number_of_axial_points,
            bearing_load_x=bearing_load_x,
            bearing_load_y=bearing_load_y,
        )
