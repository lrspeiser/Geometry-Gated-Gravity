#!/usr/bin/env python3
"""
Unit Tests for Lensing Cosmology Module

Comprehensive testing suite addressing Editor Concern D.
Tests all unit conversions, distance calculations, and edge cases.

Usage:
    python test_cosmology_units.py
    
or with pytest:
    pytest test_cosmology_units.py -v
"""

import numpy as np
import pytest
from lensing_cosmology import LensingCosmology, PhysicalConstants, get_default_cosmology


class TestPhysicalConstants:
    """Test that physical constants have correct values."""
    
    def test_speed_of_light(self):
        """Test speed of light value."""
        const = PhysicalConstants()
        assert np.isclose(const.c_km_s, 299792.458, rtol=1e-10)
    
    def test_gravitational_constant(self):
        """Test G in correct units for kpc, M_sun, km/s."""
        const = PhysicalConstants()
        # G = 4.302e-6 kpc³ M_☉⁻¹ (km/s)²
        assert np.isclose(const.G_kpc3_Msun_km2s2, 4.302e-6, rtol=1e-4)
    
    def test_arcsec_conversion(self):
        """Test arcsec per radian."""
        const = PhysicalConstants()
        assert np.isclose(const.arcsec_per_radian, 206265.0, rtol=1e-6)


class TestAngularDiameterDistance:
    """Test angular diameter distance calculations."""
    
    def setup_method(self):
        """Set up test cosmology."""
        self.cosmo = LensingCosmology(H0=70, Om0=0.3)
    
    def test_zero_redshift(self):
        """Distance at z=0 should be zero."""
        D = self.cosmo.angular_diameter_distance(0.0)
        assert D == 0.0
    
    def test_negative_redshift_raises(self):
        """Negative redshift should raise ValueError."""
        with pytest.raises(ValueError):
            self.cosmo.angular_diameter_distance(-0.1)
    
    def test_low_redshift_approximate(self):
        """At low z, D_A ≈ cz/H0."""
        z = 0.01
        D = self.cosmo.angular_diameter_distance(z)
        D_approx = (299792.458 / 70.0) * z  # Mpc
        assert np.isclose(D, D_approx, rtol=0.01)
    
    def test_typical_cluster_redshift(self):
        """Test typical cluster lens redshift."""
        z = 0.5
        D = self.cosmo.angular_diameter_distance(z)
        # Should be ~800-1200 Mpc for typical cosmology
        assert 800 < D < 1200, f"D_A(0.5) = {D} Mpc is outside expected range"
    
    def test_high_redshift_source(self):
        """Test typical source redshift."""
        z = 2.0
        D = self.cosmo.angular_diameter_distance(z)
        # Should be reasonable
        assert 1000 < D < 2000, f"D_A(2.0) = {D} Mpc is outside expected range"
    
    def test_increasing_with_redshift_early(self):
        """D_A increases with z at low redshift."""
        D1 = self.cosmo.angular_diameter_distance(0.1)
        D2 = self.cosmo.angular_diameter_distance(0.2)
        assert D2 > D1


class TestAngularDiameterDistanceZ1Z2:
    """Test angular diameter distance between two redshifts."""
    
    def setup_method(self):
        """Set up test cosmology."""
        self.cosmo = LensingCosmology(H0=70, Om0=0.3)
    
    def test_z2_must_be_greater_than_z1(self):
        """z2 must be greater than z1."""
        with pytest.raises(ValueError):
            self.cosmo.angular_diameter_distance_z1z2(0.5, 0.5)
        
        with pytest.raises(ValueError):
            self.cosmo.angular_diameter_distance_z1z2(0.5, 0.3)
    
    def test_typical_lens_source_configuration(self):
        """Test typical z_lens=0.5, z_source=2.0."""
        D_ls = self.cosmo.angular_diameter_distance_z1z2(0.5, 2.0)
        # Should be positive and reasonable
        assert 1000 < D_ls < 2500, f"D_ls(0.5, 2.0) = {D_ls} Mpc is outside expected range"
    
    def test_consistency_with_single_distances(self):
        """Check approximate consistency: D_ls ≈ D_s - D_d for low z."""
        z1, z2 = 0.1, 0.2
        D1 = self.cosmo.angular_diameter_distance(z1)
        D2 = self.cosmo.angular_diameter_distance(z2)
        D_12 = self.cosmo.angular_diameter_distance_z1z2(z1, z2)
        
        # At low z, should be approximately D2 - D1
        # (not exact due to cosmological effects)
        assert np.isclose(D_12, (D2 - D1) * (1 + z1) / (1 + z2), rtol=0.1)


class TestCriticalSurfaceDensity:
    """Test critical surface density calculations."""
    
    def setup_method(self):
        """Set up test cosmology."""
        self.cosmo = LensingCosmology(H0=70, Om0=0.3)
    
    def test_source_behind_lens_required(self):
        """Source must be behind lens."""
        with pytest.raises(ValueError):
            self.cosmo.critical_surface_density(z_lens=0.5, z_source=0.5)
        
        with pytest.raises(ValueError):
            self.cosmo.critical_surface_density(z_lens=0.5, z_source=0.3)
    
    def test_typical_values(self):
        """Test that Sigma_crit has reasonable values."""
        Sigma_crit = self.cosmo.critical_surface_density(z_lens=0.5, z_source=2.0)
        # Should be ~1-10 × 10^9 M_☉/kpc²
        assert 1e9 < Sigma_crit < 1e10, f"Σ_crit = {Sigma_crit:.2e} M_☉/kpc² is outside expected range"
    
    def test_increases_with_source_redshift(self):
        """Sigma_crit should increase with source redshift."""
        z_lens = 0.5
        Sigma1 = self.cosmo.critical_surface_density(z_lens, z_source=1.5)
        Sigma2 = self.cosmo.critical_surface_density(z_lens, z_source=2.5)
        # Greater D_s → greater Sigma_crit
        assert Sigma2 > Sigma1


class TestCoordinateConversions:
    """Test angular <-> physical coordinate conversions."""
    
    def setup_method(self):
        """Set up test cosmology."""
        self.cosmo = LensingCosmology(H0=70, Om0=0.3)
    
    def test_round_trip_conversion(self):
        """Physical → angular → physical should recover original."""
        R_kpc = 100.0
        z_lens = 0.5
        
        theta = self.cosmo.physical_to_angular(R_kpc, z_lens)
        R_back = self.cosmo.angular_to_physical(theta, z_lens)
        
        assert np.isclose(R_kpc, R_back, rtol=1e-10), \
            f"Round-trip failed: {R_kpc} → {theta}\" → {R_back} kpc"
    
    def test_angular_increases_with_redshift(self):
        """Same physical size has smaller angular size at higher z."""
        R_kpc = 100.0
        
        theta1 = self.cosmo.physical_to_angular(R_kpc, z_lens=0.3)
        theta2 = self.cosmo.physical_to_angular(R_kpc, z_lens=0.6)
        
        # Higher z → smaller angular size
        assert theta2 < theta1
    
    def test_typical_cluster_scale(self):
        """Test conversion for typical cluster scale."""
        R_kpc = 500.0  # Typical cluster scale
        z_lens = 0.5
        
        theta = self.cosmo.physical_to_angular(R_kpc, z_lens)
        
        # Should be tens of arcsec
        assert 20 < theta < 200, f"{R_kpc} kpc at z={z_lens} → {theta}\" is outside expected range"
    
    def test_zero_distance_raises(self):
        """Cannot convert at z=0."""
        with pytest.raises(ValueError):
            self.cosmo.physical_to_angular(100.0, z_lens=0.0)


class TestDeflectionAngles:
    """Test deflection angle calculations."""
    
    def setup_method(self):
        """Set up test cosmology."""
        self.cosmo = LensingCosmology(H0=70, Om0=0.3)
    
    def test_point_mass_deflection_scales_with_mass(self):
        """Deflection angle should scale linearly with mass."""
        M1 = 1e14
        M2 = 2e14
        R_kpc = 100.0
        z_lens, z_source = 0.5, 2.0
        
        alpha1 = self.cosmo.deflection_angle_point_mass(M1, R_kpc, z_lens, z_source)
        alpha2 = self.cosmo.deflection_angle_point_mass(M2, R_kpc, z_lens, z_source)
        
        assert np.isclose(alpha2 / alpha1, 2.0, rtol=1e-10)
    
    def test_point_mass_deflection_scales_with_radius(self):
        """Deflection angle should scale as 1/R."""
        M = 1e14
        R1 = 100.0
        R2 = 200.0
        z_lens, z_source = 0.5, 2.0
        
        alpha1 = self.cosmo.deflection_angle_point_mass(M, R1, z_lens, z_source)
        alpha2 = self.cosmo.deflection_angle_point_mass(M, R2, z_lens, z_source)
        
        assert np.isclose(alpha1 / alpha2, 2.0, rtol=1e-10)
    
    def test_einstein_radius_reasonable(self):
        """Einstein radius should have reasonable value."""
        M = 1e14  # Typical cluster mass
        z_lens, z_source = 0.5, 2.0
        
        theta_E = self.cosmo.einstein_radius(M, z_lens, z_source)
        
        # Should be tens of arcsec for typical cluster
        assert 10 < theta_E < 100, f"θ_E = {theta_E}\" for M={M:.1e} is outside expected range"
    
    def test_einstein_radius_consistency(self):
        """Einstein radius from formula should match deflection at R_E."""
        M = 1e14
        z_lens, z_source = 0.5, 2.0
        
        theta_E = self.cosmo.einstein_radius(M, z_lens, z_source)
        R_E = self.cosmo.angular_to_physical(theta_E, z_lens)
        
        # At R_E, deflection angle should equal theta_E
        alpha = self.cosmo.deflection_angle_point_mass(M, R_E, z_lens, z_source)
        
        assert np.isclose(alpha, theta_E, rtol=0.01), \
            f"Inconsistent: θ_E={theta_E}\", but α(R_E)={alpha}\""


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up test cosmology."""
        self.cosmo = LensingCosmology(H0=70, Om0=0.3)
    
    def test_very_high_redshift(self):
        """Test at very high redshift (z=10)."""
        D = self.cosmo.angular_diameter_distance(10.0)
        assert D > 0, "Distance at z=10 should be positive"
        assert D < 2000, f"Distance at z=10 seems too large: {D} Mpc"
    
    def test_different_cosmologies(self):
        """Test with different cosmological parameters."""
        cosmo1 = LensingCosmology(H0=67, Om0=0.32)
        cosmo2 = LensingCosmology(H0=73, Om0=0.27)
        
        D1 = cosmo1.angular_diameter_distance(1.0)
        D2 = cosmo2.angular_diameter_distance(1.0)
        
        # Should be different but both reasonable
        assert abs(D1 - D2) > 50, "Different cosmologies should give different distances"
        assert 1000 < D1 < 2000
        assert 1000 < D2 < 2000
    
    def test_consistency_across_backends(self):
        """Test that astropy and simple backends give consistent results."""
        # This test only runs if astropy is available
        try:
            cosmo_astropy = LensingCosmology(H0=70, Om0=0.3, use_astropy=True)
            cosmo_simple = LensingCosmology(H0=70, Om0=0.3, use_astropy=False)
            
            z = 0.5
            D_astropy = cosmo_astropy.angular_diameter_distance(z)
            D_simple = cosmo_simple.angular_diameter_distance(z)
            
            # Should agree to within 1%
            assert np.isclose(D_astropy, D_simple, rtol=0.01), \
                f"Backend mismatch: astropy={D_astropy}, simple={D_simple}"
        
        except ImportError:
            pytest.skip("astropy not available for backend comparison")


class TestIntegration:
    """Integration tests for complete lensing calculations."""
    
    def test_complete_lensing_calculation(self):
        """Test complete workflow from cluster properties to deflection."""
        # Set up
        cosmo = LensingCosmology(H0=70, Om0=0.3)
        
        # Cluster properties
        z_lens = 0.5
        z_source = 2.0
        M_cluster = 1e14  # M_☉
        R_core = 100.0  # kpc
        
        # 1. Compute critical density
        Sigma_crit = cosmo.critical_surface_density(z_lens, z_source)
        assert Sigma_crit > 0
        
        # 2. Convert to angular scales
        theta_core = cosmo.physical_to_angular(R_core, z_lens)
        assert theta_core > 0
        
        # 3. Compute Einstein radius
        theta_E = cosmo.einstein_radius(M_cluster, z_lens, z_source)
        assert theta_E > 0
        
        # 4. Compute deflection at core radius
        alpha = cosmo.deflection_angle_point_mass(M_cluster, R_core, z_lens, z_source)
        assert alpha > 0
        
        # Sanity check: deflection should be less than Einstein radius at R > R_E
        R_E = cosmo.angular_to_physical(theta_E, z_lens)
        if R_core > R_E:
            assert alpha < theta_E
    
    def test_default_cosmology_functions(self):
        """Test convenience functions for standard cosmologies."""
        cosmo_default = get_default_cosmology()
        assert cosmo_default.H0 == 67.4
        assert cosmo_default.Om0 == 0.315
        
        from lensing_cosmology import get_wmap_cosmology
        cosmo_wmap = get_wmap_cosmology()
        assert cosmo_wmap.H0 == 69.3
        assert cosmo_wmap.Om0 == 0.286


def run_all_tests():
    """Run all tests and report results."""
    print("="*70)
    print("LENSING COSMOLOGY UNIT TESTS")
    print("="*70)
    print()
    
    # Try to import pytest
    try:
        import pytest
        print("Running tests with pytest...")
        pytest.main([__file__, '-v', '--tb=short'])
    except ImportError:
        print("pytest not available, running manual tests...")
        print()
        
        # Manual test running
        test_classes = [
            TestPhysicalConstants,
            TestAngularDiameterDistance,
            TestAngularDiameterDistanceZ1Z2,
            TestCriticalSurfaceDensity,
            TestCoordinateConversions,
            TestDeflectionAngles,
            TestEdgeCases,
            TestIntegration,
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            print(f"\n{test_class.__name__}")
            print("-" * 70)
            
            test_instance = test_class()
            if hasattr(test_instance, 'setup_method'):
                test_instance.setup_method()
            
            # Get all test methods
            test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
            
            for method_name in test_methods:
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
        
        print()
        print("="*70)
        print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
        print("="*70)
        
        if passed_tests == total_tests:
            print("✓ ALL TESTS PASSED")
            return 0
        else:
            print(f"✗ {total_tests - passed_tests} TESTS FAILED")
            return 1


if __name__ == "__main__":
    exit(run_all_tests())
