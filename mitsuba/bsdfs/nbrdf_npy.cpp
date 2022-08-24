

#include <mitsuba/core/fresolver.h>

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf_mod.h"
#include "microfacet.h"
#include "nn.h"

MTS_NAMESPACE_BEGIN

typedef vector<vector<float>> vector2D;
typedef vector<float> vector1D;

class nbrdf_npy : public BSDF {
public:
	nbrdf_npy(const Properties &props) : BSDF(props) {
		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
				: "diffuseReflectance", Spectrum(.5f)));

		// load NN
		fs::path m_nn_basename = Thread::getThread()->getFileResolver()->resolve(props.getString("nn_basename"));
		net = Net(m_nn_basename.c_str());
		aux_brdf = new djb::merl(net);
		m_tabular = new djb::tabular(*aux_brdf, 90, true);
	}

	nbrdf_npy(Stream *stream, InstanceManager *manager) : BSDF(stream, manager) {
		configure();
	}

	~nbrdf_npy() {
		delete m_tabular;
	}

	void configure() {
		/* Verify the input parameter and fix them if necessary */
		m_components.clear();
		m_components.push_back(EDiffuseReflection | EFrontSide | 0);
		m_usesRayDifferentials = false;
		BSDF::configure();
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		djb::vec3 h(0.0, 0.0, 0.0);
		djb::vec3 d(0.0, 0.0, 0.0);
		aux_brdf->io_to_hd(i, o, &h, &d);

		float_t theta_h, phi_h;
		djb::xyz_to_theta_phi(h, &theta_h, &phi_h);
		phi_h = 0.0;
		float_t hx = sin(theta_h);
		float_t hy = 0.0;
		float_t hz = cos(theta_h);

		vector1D x = {hx, hy, hz, d.x, d.y, d.z};
		vector1D y = net.forward(x);

		return Color3(y[0], y[1], y[2]) * Frame::cosTheta(bRec.wo);
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		return m_tabular->pdf(i, o);
	}


	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);

		/* Sample the tabulated microfacet BRDF */
		djb::vec3 o = djb::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i = m_tabular->sample(sample.x, sample.y, o);

		/* Setup Mitsuba variables */
		bRec.wo = Vector(i.x, i.y, i.z);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;

		/* Side check */
		if (Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		return eval(bRec, ESolidAngle) / pdf(bRec, ESolidAngle);
	}

    Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf_, const Point2 &sample_) const {
        Spectrum res = sample(bRec, sample_);
        pdf_ = pdf(bRec, ESolidAngle);
        return res;
    }

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))
				&& (name == "reflectance" || name == "diffuseReflectance")) {
		} else {
			BSDF::addChild(name, child);
		}
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "nbrdf_npy[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	Net net;
	djb::brdf* aux_brdf;
	djb::tabular* m_tabular;
};

// ================ Hardware shader implementation ================

class fullmerl_nn_shader : public Shader { //GPU version not implemented
public:
	fullmerl_nn_shader(Renderer *renderer, const Texture *reflectance)
		: Shader(renderer, EBSDFShader), m_reflectance(reflectance) {
		m_reflectanceShader = renderer->registerShaderForResource(m_reflectance.get());
	}

	bool isComplete() const {
		return m_reflectanceShader.get() != NULL;
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_reflectance.get());
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_reflectanceShader.get());
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return " << depNames[0] << "(uv) * inv_pi * cosTheta(wo);" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return " << evalName << "(uv, wi, wo);" << endl
			<< "}" << endl;
	}

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	ref<Shader> m_reflectanceShader;
};

Shader *nbrdf_npy::createShader(Renderer *renderer) const {
	return new fullmerl_nn_shader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(fullmerl_nn_shader, false, Shader)
MTS_IMPLEMENT_CLASS_S(nbrdf_npy, false, BSDF)
MTS_EXPORT_PLUGIN(nbrdf_npy, "nbrdf_npy BRDF")
MTS_NAMESPACE_END
