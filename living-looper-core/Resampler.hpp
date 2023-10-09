#pragma once
#include <cmath>
#include <vector>
#include <iostream>

float sinc(float x){
    return x==0 ? 1.0 : std::sin(x*M_PI) / (x*M_PI);
}

// TODO: special case when rate_in == rate_out
class Resampler {
public:
    long m_rate_in;
    long m_rate_out;
    int m_lanczos_n;
    long m_lanczos_rate;
    int m_filt_len;
    int m_head;
    //stored as # in samples
    long m_next_in;
    long m_last_in;
    // stored as # out samples 
    long m_next_out;
    long m_last_out;

    float delay;

    std::vector<float> m_values;
    std::vector<long> m_times;

    // Resampler(){}

    Resampler(int rate_in, int rate_out, int lanczos_n){
        std::cout << "resampler: " << rate_in << " to " << rate_out << std::endl;

        m_rate_in = rate_in;
        m_rate_out = rate_out;
        m_lanczos_n = lanczos_n;
        m_lanczos_rate = std::min(rate_in, rate_out);

        // TODO: this should result in a m_filt_len=1 when lanczos_n==0 ...
        // could then just set to 0 if sample rates are the same 
        m_filt_len = int(std::ceil(
            rate_in / m_lanczos_rate * (m_lanczos_n + 1) * 2
        ));

        m_values = std::vector<float>(m_filt_len);
        m_times = std::vector<long>(m_filt_len);

        m_next_in = 0;
        m_last_in = -1;
        m_next_out = 0;
        m_last_out = -1;

        delay = float(m_lanczos_n) / m_lanczos_rate;
    }
    float get_dt(long t_in, long t_out){
        // std::cout << t_in << " " << t_out << std::endl;
        // std::cout << t_out * m_rate_in - t_in * m_rate_out << " " << m_rate_out * m_rate_in << std::endl;
        return 
            float(t_out * m_rate_in - t_in * m_rate_out)
            / (m_rate_out * m_rate_in);
    }
    float filter(float t){
        const float t_center = t - delay; // in seconds
        const float t_scale = t_center * m_lanczos_rate; // in samples at lanczos rate
        const float w = 
            sinc(t_scale/m_lanczos_n) 
            * ((std::fabs(t_scale) <= m_lanczos_n) ? 1.0f : 0.0f);

        // std::cout << t << " " << delay << " " << t_center << " " << t_scale << " " << w << std::endl;

        return w * sinc(t_scale);
    }
    float read(){
        // DEBUG
        // m_last_out = m_next_out;m_next_out += 1; return m_values[0];

        float num = 0;
        float denom = 1e-15;
        long t;
        for (int i=0; i<m_filt_len; i++){
            const auto t = m_times[i];
            const auto v = m_values[i];
            const auto dt = get_dt(t, m_next_out);
            const auto w = filter(dt);
            num += w * v;
            denom += w;
            // std::cout << "t " << t << "v " << v << "w " << w << std::endl;
        }
        m_last_out = m_next_out;
        m_next_out += 1;

        // std::cout << "read " << num << "/" << denom << std::endl;

        return num / denom;
    }
    void write(float x){
        //DEBUG
        // m_last_in = m_next_in; m_next_in += 1; m_values[0] = x; return;

        // std::cout << "x " << x << " m_head " << m_head << " m_filt_len " << m_filt_len << std::endl;
        m_values[m_head] = x;
        m_times[m_head] = m_next_in;

        m_last_in = m_next_in;
        m_next_in += 1;

        m_head += 1;
        m_head %= m_filt_len;
    }
    bool pending(){
        return 
            (m_last_in >= 0) && 
            (m_next_out * m_rate_in <= m_last_in * m_rate_out);
    }
};